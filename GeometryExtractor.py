import os
import argparse

import cv2
import numpy as np
import open3d as o3d
import sys

class segmentor(object):
    def cropbox(self,cloud,minim,maxim,dim):
        dim_len = cloud.shape[-1]
        cloud_proc = np.squeeze(cloud[np.argwhere(cloud[:,dim]>minim),:])
        cloud_proc = np.squeeze(cloud_proc[np.argwhere(cloud_proc[:,dim]<maxim),:])
        return cloud_proc

    def PlaneFinder(self,points):
        """
        Input: 3d Points
        Output: Plane Equation and mean of points [a,b,c,d]
        Explanation:
        Obtain the plane equation by:
        1.Compute Covariance matrix (x-avg_x)*(x-avg_x).transpose()
        2.Eigenvector with smallest eigenvalue is normal to plane [a,b,c].
        3. d = dot([normal,1], point_in_plane)
        plane_equation = [a,b,c,d]
        """
        points = points[:,:3]
        # print(points)
        points_len = points.shape[0]
        point_avg = np.sum(points,axis=0)/points_len
        point_moved = (points-point_avg).reshape(-1,3)
        Covar = np.dot(point_moved.T,point_moved)/points_len
        # print(Covar)
        _,eig_vector = np.linalg.eig(Covar)
        n_plane = eig_vector[:,-1] #Obtain smallest EigenValue
        d_plane = -np.dot(n_plane,point_avg)
        plane_eq = np.vstack([n_plane.reshape(-1,1),d_plane])
        return plane_eq,point_avg

    def ProjectToPlane(self,plane,points):
        '''
        Point_On_Plane = Point - Perpendicular_Distance_to_Plane*Normal
        '''
        point_len = points.shape[0]
        points[:,-1] = np.ones(point_len)
        #Getting Perpendicular distance of each point to plane
        PerDistance = np.sum(plane.T*points,axis=1)
        trans = (PerDistance*plane[:3]).T 
        points[:,:3] = points[:,:3] - trans 
        return points

    def LineFinder(self,points):
        points = points[:,:3]
        points_len = points.shape[0]
        point_avg = np.sum(points,axis=0)/points_len
        point_moved = (points-point_avg).reshape(-1,3)
        Covar = np.dot(point_moved.T,point_moved)/points_len
        _,eig_vector = np.linalg.eig(Covar)
        dir_line = eig_vector[:,0].reshape(-1,1) #Obtain Largest EigenValue
        return dir_line.reshape(-1,),point_avg

    def ProjectToLine(self,pointA,pointB,inquiry):
        AI = inquiry-pointA
        AB = pointB-pointA
        proj_point  = pointA + np.dot(AI.reshape(-1,),AB.reshape(-1,))/(np.linalg.norm(AB)**2) *AB
        return proj_point

    def RANSACLargestPlane(self,points,max_iter,sample_sz):
        point_len = points.shape[0]
        min_points = int(point_len*0.9)
        best_inlier = 0
        dist_thresh = 0.7
        itr = 0
        ones = np.ones(points.shape[0]).reshape(-1,1)
        points = np.hstack([points, ones])
        while(best_inlier<min_points):
            sample_ind = (point_len*np.random.rand(sample_sz)).astype(int)
            sample_points = points[sample_ind,:]
            proposed_plane,mean_point = self.PlaneFinder(sample_points)
            dist = np.abs(np.sum(proposed_plane.T*points,axis=1))
            inliers = dist<dist_thresh
            inliers_ind = np.argwhere(inliers)
            inliers_num = np.sum(inliers)
            if(inliers_num>best_inlier):
                best_plane = proposed_plane
                best_ind = inliers_ind
                best_inlier = inliers_num
                best_mean_point = mean_point.reshape(-1,1)
            if(itr>max_iter):
                break
            itr = itr+1
        return best_plane,best_ind,best_mean_point

    def spherical_coord(self,cloud):
        '''r = len*sin(Phi)
           z = len*cos(Phi) 
           => Phi/Pitch = atan2(r,z) '''
        x,y,z,_ = cloud.T
        r = np.sqrt(x**2+y**2)
        beam_id = np.unique(np.round(np.rad2deg(np.arctan2(r,z))).astype(int))
        beam_index = np.round(np.rad2deg(np.arctan2(r,z))).astype(int).reshape(-1,1)
        cloud_ind = np.hstack([cloud,beam_index])
        scan_line = [cloud[np.argwhere(cloud_ind[:,-1] == beam).reshape(-1,),:] for beam in beam_id]
        # scan_ind = [np.argwhere(cloud_ind[:,-1] == beam).reshape(-1,) for beam in beam_id]
        return scan_line

    def RANSAC_line(self,points,max_iter):
        #Idea: Do NULL space rep: Build Matrix [Point1;Point2;Point_being_evaluated], if rank is still 2 then implies it is still part of the line.
        #We DSTACK this, swapaxes(0,2) S.T shape = [Depth,Matrix Row, Matrix Col], and then np.linalg.matrix_rank, which ever has almost 0 determinant -> inlier.

        point_len = points.shape[0]
        min_points = int(point_len*0.7)
        points[:,-1] = np.ones(point_len)
        best_inlier = 0
        dist_thresh = 0.05
        itr = 0
        while(best_inlier<min_points):
            sample_ind = (point_len*np.random.rand(2)).astype(int)
            W = np.expand_dims(np.vstack([points[sample_ind,:3],np.array([0,0,0])]),axis=0)
            proposed_line = np.repeat(W,point_len,axis=0)
            proposed_line[:,-1,:] = points[:,:3]
            dist = np.abs(np.linalg.det(proposed_line))
            #TODO: adapt lines below
            inliers = dist<dist_thresh
            inliers_ind = np.argwhere(inliers)
            inliers_num = np.sum(inliers)
            if(inliers_num>best_inlier):
                best_line = proposed_line
                best_ind = inliers_ind
                best_inlier = inliers_num
                print("line",best_inlier,point_len)
            if(itr>max_iter):
                break
            itr = itr+1
        return best_line,best_ind
