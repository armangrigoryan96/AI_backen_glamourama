
from typing import List, Dict
import numpy as np
import json
import trimesh
import torch
import smplx
from pprint import pprint
import os
from scipy.spatial import ConvexHull
import plotly
import plotly.graph_objects as go
import plotly.express as px
from measurement_definitions import *
from utils import *
import panel as pn
from scipy.spatial import distance
import argparse



def set_shape(model, shape_coefs):
    '''
    Set shape of SMPL model.
    :param model: SMPL body model
    :param shape_coefs: torch.tensor dim (10,)

    Return
    shaped SMPL body model
    '''
    shape_coefs = shape_coefs.to(torch.float32)
    return model(betas=shape_coefs, return_verts=True)

def create_model(smpl_path, gender, num_coefs=10):
    '''
    Create SMPL body model
    :param smpl_path: str of location to SMPL .pkl models
    :param gender: str of gender: MALE or FEMALE or NEUTRAL
    :param num_coefs: int of number of SMPL shape coefficients
                      requires the model with num_coefs in smpl_path
    
    Return:
    :param SMPL body model
    '''
    
    body_pose = torch.zeros((1, (SMPL_NUM_KPTS-1) * 3))
    if smpl_path.split("/")[-1] == "smpl":
        smpl_path = os.path.dirname(smpl_path)
    
    return smplx.create(smpl_path, 
                        model_type="smpl",
                        gender=gender, 
                        use_face_contour=False,
                        num_betas=num_coefs,
                        body_pose=body_pose,
                        ext='pkl')

def get_SMPL_joint_regressor(smpl_path):
    '''
    Extract joint regressor from SMPL body model
    :param smpl_path: str of location to SMPL .pkl models
    
    Return:
    :param model.J_regressor: torch.tensor (23,6890) used to 
                              multiply with body model to get 
                              joint locations
    '''

    if smpl_path.split("/")[-1] == "smpl":
        smpl_path = os.path.dirname(smpl_path)

    model = smplx.create(smpl_path, 
                    model_type="smpl",
                    gender="MALE", 
                    use_face_contour=False,
                    num_betas=10,
                    body_pose=torch.zeros((1, 23 * 3)),
                    ext='pkl')
    return model.J_regressor


class MeasureSMPL():
    '''
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices. 

    All the measurements are expressed in cm.
    '''

    def __init__(self,
                 smpl_path: str
                ):
        self.smpl_path = smpl_path

        self.verts = None
        self.faces = smplx.SMPL(os.path.join(self.smpl_path,"smpl")).faces
        self.joints = None
        self.gender = None

        self.face_segmentation = load_face_segmentation(self.smpl_path)

        self.measurements = {}
        self.height_normalized_measurements = {}
        self.labeled_measurements = {}
        self.height_normalized_labeled_measurements = {}
        self.labels2names = {}
        self.landmarks = LANDMARK_INDICES
        self.measurement_types = MeasurementDefinitions().measurement_types
        self.length_definitions = MeasurementDefinitions().LENGTHS
        self.circumf_definitions = MeasurementDefinitions().CIRCUMFERENCES
        self.circumf_2_bodypart = MeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self.cached_visualizations = {"LENGTHS":{}, "CIRCUMFERENCES":{}}
        self.all_possible_measurements = MeasurementDefinitions().possible_measurements

    def from_verts(self,
                   verts: torch.tensor):
        '''
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        '''        

        assert verts.shape == torch.Size([6890,3]), "verts need to be of dimension (6890,3)"

        joint_regressor = get_SMPL_joint_regressor(self.smpl_path)
        joints = torch.matmul(joint_regressor.float(), verts.float())
        self.joints = joints.numpy()
        self.verts = verts.numpy()

    def from_smpl(self,
                  gender: str,
                  shape: torch.tensor):
        '''
        Construct body model from given gender and shape params 
        of SMPl model.
        :param gender: str, MALE or FEMALE or NEUTRAL
        :param shape: torch.tensor, (1,10) beta parameters
                                    for SMPL model
        '''  
        model = create_model(self.smpl_path,gender)    
        model_output = set_shape(model, shape)
        
        self.verts = model_output.vertices.detach().cpu().numpy().squeeze()
        self.joints = model_output.joints.squeeze().detach().cpu().numpy()
        self.gender = gender



    @staticmethod
    def create_mesh_plot(verts: np.ndarray, faces: np.ndarray):
        '''
        Visualize smpl body mesh.
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices

        Return:
        plotly Mesh3d object for plotting
        '''
        mesh_plot = go.Mesh3d(
                            x=verts[:,0],
                            y=verts[:,1],
                            z=verts[:,2],
                            color='rgb(232, 190, 172)',
                            hovertemplate ='<i>Index</i>: %{text}',
                            text = [i for i in range(verts.shape[0])],
                            # i, j and k give the vertices of triangles
                            i=faces[:,0],
                            j=faces[:,1],
                            k=faces[:,2],
                            opacity=0.6,
                            name='body',
                            )
        return mesh_plot
        
    @staticmethod
    def create_joint_plot(joints: np.ndarray):

        return go.Scatter3d(x = joints[:,0],
                            y = joints[:,1], 
                            z = joints[:,2], 
                            mode='markers',
                            marker=dict(size=8,
                                        color='rgb(232, 190, 172)',
                                        opacity=1,
                                        symbol="cross"
                                        ),
                            name="joints"
                                )
    
    @staticmethod
    def create_wireframe_plot(verts: np.ndarray,faces: np.ndarray):
        '''
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.
        
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        '''
        i=faces[:,0]
        j=faces[:,1]
        k=faces[:,2]

        triangles = np.vstack((i,j,k)).T

        x=verts[:,0]
        y=verts[:,1]
        z=verts[:,2]

        vertices = np.vstack((x,y,z)).T
        tri_points = vertices[triangles]

        #extract the lists of x, y, z coordinates of the triangle 
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])

        # return Xe, Ye, Ze 
        wireframe = go.Scatter3d(
                        x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='wireframe',
                        line=dict(color= 'rgb(232, 190, 172)', width=1)
                        )
        return wireframe
        # rgb(70, 70, 70) (232, 190, 172)
    def create_landmarks_plot(self,
                              landmark_names: List[str], 
                              verts: np.ndarray
                              ) -> List[plotly.graph_objs.Scatter3d]:
        '''
        Visualize landmarks from landmark_names list
        :param landmark_names: List[str] of landmark names to visualize

        Return
        :param plots: list of plotly objects to plot
        '''

        plots = []

        landmark_colors = dict(zip(self.landmarks.keys(),
                                px.colors.qualitative.Alphabet))

        for lm_name in landmark_names:
            if lm_name not in self.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            lm_index = self.landmarks[lm_name]
            if isinstance(lm_index,tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index] 

            plot = go.Scatter3d(x = [lm[0]],
                                y = [lm[1]], 
                                z = [lm[2]], 
                                mode='markers',
                                marker=dict(size=8,
                                            color=landmark_colors[lm_name],
                                            opacity=0.5,
                                            ),
                               name=lm_name
                                )

            plots.append(plot)

        return plots

    def create_measurement_length_plot(self, 
                                       measurement_name: str,
                                       verts: np.ndarray,
                                       color: str
                                       ):
        '''
        Create length measurement plot.
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''
        
        measurement_landmarks_inds = self.length_definitions[measurement_name]

        segments = {"x":[],"y":[],"z":[]}
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                lm_tnp = (verts[measurement_landmarks_inds[i][0]] + 
                          verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm_tnp = verts[measurement_landmarks_inds[i]]
            segments["x"].append(lm_tnp[0])
            segments["y"].append(lm_tnp[1])
            segments["z"].append(lm_tnp[2])
        for ax in ["x","y","z"]:
            segments[ax].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(x=segments["x"], 
                                    y=segments["y"], 
                                    z=segments["z"],
                                    marker=dict(
                                            size=4,
                                            color="rgba(0,0,0,0)",
                                        ),
                                        line=dict(
                                            color=color,
                                            width=10),
                                        name=m_viz_name
                                        )
        
    def create_measurement_circumference_plot(self,
                                              measurement_name: str,
                                              verts: np.ndarray,
                                              faces: np.ndarray,
                                              color: str):
        '''
        Create circumference measurement plot
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        '''

        circumf_landmarks = self.circumf_definitions[measurement_name]["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]
        
        plane_origin = np.mean(verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                        plane_normal=plane_normal, 
                        plane_origin=plane_origin, 
                        return_faces=True) # (N, 2, 3), (N,)

        slice_segments = self._filter_body_part_slices(slice_segments,
                                                      sliced_faces,
                                                      measurement_name)
        
        slice_segments_hull = self._circumf_convex_hull(slice_segments)
        
        
        draw_segments = {"x":[],"y":[],"z":[]}
        map_ax = {0:"x",1:"y",2:"z"}

        for i in range(slice_segments_hull.shape[0]):
            for j in range(3):
                draw_segments[map_ax[j]].append(slice_segments_hull[i,0,j])
                draw_segments[map_ax[j]].append(slice_segments_hull[i,1,j])
                draw_segments[map_ax[j]].append(None)

        if measurement_name in self.measurements:
            m_viz_name = f"{measurement_name}: {self.measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(
                            x=draw_segments["x"],
                            y=draw_segments["y"],
                            z=draw_segments["z"],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=10),
                            name=m_viz_name
                                )

    pn.extension("plotly")
    pn.config.sizing_mode = "stretch_width"
    def visualize(self, 
                  measurements: List[str] = [], 
                  landmarks: List[str] = [],
                  title="Measurement visualization"):
        '''
        Visualize the SMPL mesh with measurements and landmarks.

        :param measurements: List[str], list of strings with measurement names
        :param landmarks: List[str], list of strings with landmark names
        :param title: str, title of plot
        '''

        if self.verts is None:
            print("Model has not been defined. \
                  Visualizing on default SMPL male model")
            model = create_model(self.smpl_path, "MALE", num_coefs=10)
            shape = torch.zeros((1, 10), dtype=torch.float32)
            model_output = set_shape(model, shape)
            
            verts = model_output.vertices.detach().cpu().numpy().squeeze()
            faces = model.faces.squeeze()
        else:
            verts = self.verts
            faces = self.faces 

        if measurements == []:
            measurement_names = self.all_possible_measurements

        if landmarks == []:
            landmark_names = list(self.landmarks.keys())

        # visualize model mesh
        fig = go.Figure(data=
            go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="red", width=5),
            )
        )
        mesh_plot = self.create_mesh_plot(verts, faces)
        fig.add_trace(mesh_plot)

        # visualize joints
        joint_plot = self.create_joint_plot(self.joints)
        fig.add_trace(joint_plot)


        # visualize wireframe
        wireframe_plot = self.create_wireframe_plot(verts, faces)
        fig.add_trace(wireframe_plot)


        # visualize landmarks
        if "LANDMARKS" in self.cached_visualizations.keys():
            fig.add_traces(list(self.cached_visualizations["LANDMARKS"].values()))
        else:
            landmarks_plot = self.create_landmarks_plot(landmark_names, verts)
            self.cached_visualizations["LANDMARKS"] = {landmark_names[i]: landmarks_plot[i] 
                                                    for i in range(len(landmark_names))}
            fig.add_traces(landmarks_plot)
        

        # visualize measurements
        measurement_colors = dict(zip(self.measurement_types.keys(),
                                  px.colors.qualitative.Alphabet))

        for m_name in measurement_names:
            if m_name not in self.measurement_types.keys():
                print(f"Measurement {m_name} not defined.")
                pass

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                if m_name in self.cached_visualizations["LENGTHS"]:
                    measurement_plot = self.cached_visualizations["LENGTHS"][m_name]
                else:
                    measurement_plot = self.create_measurement_length_plot(measurement_name=m_name,
                                                                        verts=verts,
                                                                        color=measurement_colors[m_name])
                    self.cached_visualizations["LENGTHS"][m_name] = measurement_plot
     
            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                if m_name in self.cached_visualizations["CIRCUMFERENCES"]:
                    measurement_plot = self.cached_visualizations["CIRCUMFERENCES"][m_name]
                else:
                    measurement_plot = self.create_measurement_circumference_plot(measurement_name=m_name,
                                                                                verts=verts,
                                                                                faces=faces,
                                                                                color=measurement_colors[m_name])
                    self.cached_visualizations["CIRCUMFERENCES"][m_name] = measurement_plot
            
            fig.add_trace(measurement_plot)
        fig.update_layout(scene_aspectmode='data',
                            width=1000, height=700,
                            title=title,
                            )        
        plot_panel = pn.pane.Plotly(fig, config={"responsive": True}, sizing_mode="stretch_both")

        click_data = []
        @pn.depends(plot_panel.param.click_data, watch=True)
        def update_click_data(new_click_data):
            if new_click_data is not None:
                click_data.append(new_click_data)
                if len(click_data)%2==0:
                    point1 = click_data[-2]['points'][0]['x'], click_data[-2]['points'][0]['y'], click_data[-2]['points'][0]['z']
                    point2 = click_data[-1]['points'][0]['x'], click_data[-1]['points'][0]['y'], click_data[-1]['points'][0]['z']
                    fig.data[0].x = [point1[0], point2[0]]
                    fig.data[0].y = [point1[1], point2[1]]
                    fig.data[0].z = [point1[2], point2[2]]
                else:
                    fig.data[0].x = []
                    fig.data[0].y = []
                    fig.data[0].z = []
                fig.add_trace(go.Scatter3d(
                    x=[new_click_data['points'][0]['x']],
                    y=[new_click_data['points'][0]['y']],
                    z=[new_click_data['points'][0]['z']],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='blue'
                        
                    )
                ))
                # Remove the trace representing the clicked point
                if len(fig.data)>2:
                    fig.data = fig.data[:-1]

        @pn.depends(plot_panel.param.click_data)
        def calculate_distance(new_click_data):
            if len(click_data) >= 2:
                point1 = click_data[-2]['points'][0]['x'], click_data[-2]['points'][0]['y'], click_data[-2]['points'][0]['z']
                point2 = click_data[-1]['points'][0]['x'], click_data[-1]['points'][0]['y'], click_data[-1]['points'][0]['z']
                dist = distance.euclidean(point1, point2)
                return f'<span style="font-size: 16px; color: blue;">Distance between last two points: {dist*100:.2f}Cm</span>'
            else:
                return '<span style="font-size: 14px; color: red;">Click on two points to calculate distance.</span>'

        app = pn.Column(plot_panel, calculate_distance, )

        app.servable()
        app.show()
        



    def measure(self, 
                measurement_names: List[str]
                ):
        '''
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        '''

        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                print(f"Measurement {m_name} not defined.")
                pass

            if m_name in self.measurements:
                pass

            if self.measurement_types[m_name] == MeasurementType().LENGTH:

                value = self.measure_length(m_name)
                self.measurements[m_name] = value

            elif self.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                value = self.measure_circumference(m_name)
                self.measurements[m_name] = value
    
            else:
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        '''
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        '''

        measurement_landmarks_inds = self.length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (self.verts[measurement_landmarks_inds[i][0]] + 
                          self.verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm = self.verts[measurement_landmarks_inds[i]]
            
            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None,...]

        return self._get_dist(landmark_points)

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        '''
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i 
        of 3D vertices (i,0,:) and (i,1,:) 
        :param verts: np.ndarray (N,2,3) - vertices used 
                        to find distances
        
        Returns:
        :param dist: float, sumed distances between vertices
        '''

        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100 # convert to cm
        return distance_cm
    
    def measure_circumference(self, 
                              measurement_name: str, 
                              ):
        '''
        Measure circumferences. Circumferences are defined with 
        landmarks and joints - the measurement is found by cutting the 
        SMPL model with the  plane defined by a point (landmark point) and 
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        measurement_definition = self.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [self.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]
        
        plane_origin = np.mean(self.verts[circumf_landmark_indices,:],axis=0)
        plane_normal = self.joints[circumf_n1,:] - self.joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        # new version            
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                                plane_normal=plane_normal, 
                                plane_origin=plane_origin, 
                                return_faces=True) # (N, 2, 3), (N,)
        
        slice_segments = self._filter_body_part_slices(slice_segments,
                                                    sliced_faces,
                                                    measurement_name)
        
        slice_segments_hull = self._circumf_convex_hull(slice_segments)

        return self._get_dist(slice_segments_hull)
    
    def _filter_body_part_slices(self, 
                             slice_segments:np.ndarray, 
                             sliced_faces:np.ndarray,
                             measurement_name: str
                            ):
        '''
        Remove segments that are not in the appropriate body part 
        for the given measurement.
        :param slice_segments: np.ndarray - (N,2,3) for N segments 
                                            represented as two 3D points
        :param sliced_faces: np.ndarray - (N,) representing the indices of the
                                            faces
        :param measurement_name: str - name of the measurement

        Return:
        :param slice_segments: np.ndarray (K,2,3) where K < N, for K segments 
                                represented as two 3D points that are in the 
                                appropriate body part
        '''

        if measurement_name in self.circumf_2_bodypart.keys():

            body_parts = self.circumf_2_bodypart[measurement_name]

            if isinstance(body_parts,list):
                body_part_faces = [face_index for body_part in body_parts 
                                    for face_index in self.face_segmentation[body_part]]
            else:
                body_part_faces = self.face_segmentation[body_parts]

            N_sliced_faces = sliced_faces.shape[0]

            keep_segments = []
            for i in range(N_sliced_faces):
                if sliced_faces[i] in body_part_faces:
                    keep_segments.append(i)

            return slice_segments[keep_segments]

        else:
            return slice_segments

    @staticmethod
    def _circumf_convex_hull(slice_segments: np.ndarray):
        '''
        Cretes convex hull from 3D points
        :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

        Returns:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                    that form the convex hull
        '''

        # stack all points in N x 3 array
        merged_segment_points = np.concatenate(slice_segments)
        unique_segment_points = np.unique(merged_segment_points,
                                            axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        redundant_plane_coord = np.argmin(np.max(unique_segment_points,axis=0) - 
                                            np.min(unique_segment_points,axis=0) )
        non_redundant_coords = [x for x in range(3) if x!=redundant_plane_coord]

        # create convex hull
        hull = ConvexHull(unique_segment_points[:,non_redundant_coords])
        segment_point_hull_inds = hull.simplices.reshape(-1)

        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        slice_segments_hull = slice_segments_hull.reshape(-1,2,3)

        return slice_segments_hull

    def height_normalize_measurements(self, new_height: float):
        ''' 
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.
        
        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of 
                {measurement:value} pairs with 
                height measurement = new_height, and other measurements
                scaled accordingly
        '''
        if self.measurements != {}:
            old_height = self.measurements["height"]
            print(old_height)
            for m_name, m_value in self.measurements.items():
                norm_value = (m_value / old_height) * new_height
                self.measurements[m_name] = norm_value
                #self.measurements["height"] = new_height
                #print(norm_value)
            print(self.labeled_measurements)
            if self.labeled_measurements != {}:
                # old_height = self.measurements["height"]
                for m_name, m_value in self.labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self.labeled_measurements[m_name] = norm_value
            print(self.labeled_measurements)
                    #print(norm_value)

    def label_measurements(self,set_measurement_labels: Dict[str, str]):
        '''
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!
        
        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        '''

        if self.labeled_measurements != {}:
            print("Overwriting old labels")

        self.labeled_measurements = {}
        self.labels2names = {}

        for set_label, set_name in set_measurement_labels.items():
            
            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                pass

            if set_name not in self.measurements.keys():
                self.measure([set_name])

            self.labeled_measurements[set_label] = self.measurements[set_name]
            self.labels2names[set_label] = set_name

parser = argparse.ArgumentParser(description='Measurements')
parser.add_argument('-ht', '--height', type=str, default=170.5, help='Custome height of body')

args=parser.parse_args()

if __name__ == "__main__":
    scan_folder = "./output/"
    # scan_files = os.listdir(scan_folder)
    scan_files = [file for file in os.listdir(scan_folder) if file.endswith(".obj")]
    # scan_path = "./data/SMPL/smpl/test2.obj"
    # scan_path = "./smplx/output/test2.obj"
    for file in scan_files:
        file_path = os.path.join(scan_folder, file) 
        verts = trimesh.load(file_path, process=False).vertices 
        verts_tensor = torch.from_numpy(verts) 

        #verts = verts_tensor.float()
        #print(verts.shape) 

    # from measure import MeasureSMPL
    # from measurement_definitions import MeasurementDefinitions, STANDARD_LABELS

    smpl_path = "./data/SMPL"
    measurer = MeasureSMPL(smpl_path=smpl_path)

    measurer.from_verts(verts=verts_tensor)

    # smpl_path = "E:/Github Projects/SMPL-Anthropometry/data/SMPL"
    # measurer = MeasureSMPL(smpl_path=smpl_path)

    # betas = torch.zeros((1, 10), dtype=torch.float32)
    # measurer.from_smpl(gender="FEMALE", shape=betas)

    measurement_names = MeasurementDefinitions.possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)
    new_height = float(args.height)
    measurer.height_normalize_measurements(new_height)
    print("Measurements")
    pprint(measurer.measurements)

    
    print("Labeled measurements")
    pprint(measurer.labeled_measurements)

    measurer.visualize()