import cv2
import numpy as np
import pandas as pd
from glob import glob
from aruco_detector import *
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource as ls
from matplotlib import cbook, cm
import os




class Experiment():
    def __init__(self, image_path, ground_truth, cameraID):
        self.ground_truth = ground_truth
        self.detector = ArUcoDetector(cameraID = cameraID)
        self.image_path = image_path
        self.aruco_positions = []
        self.aruco_angles = []

    def run(self, experiment_name):
        self.detector.loadParams()

        if (experiment_name == 'detectionExperiment'):
            self.detectionExperiment()


    def getErrors(self, experiment_data, irl_data):
        pass

    def detectionExperiment(self):
        self.getGT()

        report = pd.DataFrame({'XC' : [], "YC" : [], "ZC" : [],  "TERROR" : [], "RERROR" : [], "PERROR" : [], "YERROR" : []})

        file_name = glob(self.image_path + "/*jpg")
        file_name.sort(key=lambda x: os.path.getctime(x), reverse = True)
        for i, file in enumerate(file_name):

            x, z = map(int, os.path.splitext(os.path.basename(file))[0].split('_'))

            img = cv2.imread(file)
            tvec, rvec = self.detector.detect(img)

            # to suitable forms of rvec and tvec
            tvectmp = []
            rvectmp = []
            for el_tvec, el_rvec in zip(tvec[0], rvec):
                tvectmp.append(el_tvec[0])
                rvectmp.append(el_rvec[0])
            tvec = tvectmp
            rvec = rvectmp

            print(file, (x, 0, z))
            # calculating translation error
            translation_error = translationError(tvec, (x, 0, z))

            # calculating rotation error
            rotation_error = rotationError(rvec, self.aruco_angles[i])


            # writing into the log and to excel file
            report.loc[len(report)] = [tvec[0], tvec[1], tvec[2],  translation_error, rotation_error[0], rotation_error[1], rotation_error[2]]

        report.to_excel('./report/report_positions.xlsx', index=False)

    def plotResults(self, axis, error):
        res_data = pd.read_excel('./report/report_positions.xlsx')
        x_ax = res_data[axis]
        y_ax = res_data[error]
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set background style
        plt.style.use('default')  # Reset to default style
        fig.patch.set_facecolor('#f8f9fa')  # Light gray background for figure
        ax.set_facecolor('#ffffff')  # White background for plot area

        # Plot the data with nice styling
        ax.scatter(x_ax, y_ax,
                color='#2E86AB',
                linewidth=2.5,
                marker='o',
                alpha=0.8,
                label='Sine Wave with Noise')

        # Set axis labels with nice formatting
        ax.set_xlabel(axis,
                      fontsize=14,
                      fontweight='bold',
                      color='#2D3047',
                      labelpad=12)

        ax.set_ylabel(error,
                      fontsize=14,
                      fontweight='bold',
                      color='#2D3047',
                      labelpad=12)

        # Set title
        ax.set_title('Report',
                     fontsize=16,
                     fontweight='bold',
                     color='#2D3047',
                     pad=20)

        # Customize ticks
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=12,
                       color='#6c757d',
                       labelcolor='#2D3047')

        # Add grid for better readability
        ax.grid(True,
                alpha=0.3,
                color='#6c757d',
                linestyle='--',
                linewidth=0.8)

        # Add legend
        ax.legend(loc='upper right',
                  fontsize=12,
                  framealpha=0.9,
                  facecolor='#ffffff',
                  edgecolor='#dee2e6')

        # Set nice spine colors
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#adb5bd')
        ax.spines['bottom'].set_color('#adb5bd')

        # Add some padding around the plot
        ax.margins(x=0.05, y=0.1)

        # Optional: Add annotation for specific points

        # Adjust layout to prevent clipping
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Optional: Save the plot
        # plt.savefig('nice_plot.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')

    def plot2Axis(self):
        res_data = pd.read_excel('./report/report_positions.xlsx')
        x = res_data['XC']
        y = res_data['ZC']
        z = np.array(res_data['TERROR'])
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        from scipy.interpolate import griddata

        # Create a grid for interpolation
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate z values
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

        # Plot the surface
        surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8)
        ax.scatter(x[z > 100], y[z > 100], z[z > 100], color='red', s=50, alpha=1.0, label='Error > 10 cm')

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Set labels
        ax.set_xlabel('XC')
        ax.set_ylabel('YC')
        ax.set_zlabel('TERROR')
        ax.set_title('Position Error Surface')

        plt.show()





    def getGT(self):
        print("LOADING GROUND TRUTH")
        data = pd.read_excel(self.ground_truth)
        x_datc = data['XC']
        y_datc = data['YC']
        z_datc = data['ZC']
        r_dat = data['R']
        p_dat = data['P']
        y_dat = data['Y']
        positions = []
        angles = []
        for i in range (len(x_datc)):
            positions.append((x_datc[i], y_datc[i], z_datc[i]))
            angles.append((r_dat[i], p_dat[i], y_dat[i]))
        self.aruco_positions = positions
        self.aruco_angles = angles
        print("GT is ready")
        print("")



def translationError(vec_exp, vec_real):
    err = 0
    '''for c_exp, c_real in zip(vec_exp, vec_real):
        err += (c_exp - c_real) ** 2
    print(vec_exp, vec_real, np.sqrt(err))'''
    err = (vec_exp[2] - vec_real[2]) ** 2 + (vec_exp[0] - vec_real[0]) ** 2
    return np.sqrt(err)

def rotationError(rotvec, angles_real):
    rotation_exp = R.from_rotvec(rotvec)
    quat_exp = rotation_exp.as_quat()
    rotation_real = R.from_euler('zyx', angles_real)
    quat_real = rotation_real.as_quat()
    quat_error = quat_real * quat_exp.conjugate()
    rotation_error = R.from_quat(quat_error)
    return(rotation_error.as_euler('zyx'))

