import cv2
import numpy as np
import matplotlib.pyplot as plt


class DLT():
    def __init__(self,x, y, z, u, v, nrRefPoints):
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v
        self.nrRefPoints = nrRefPoints
        self.splitData()


    def splitData(self):
        """
        Split the data into training and testing sets based on the number of reference points
        """
        self.xTrain = self.x[:self.nrRefPoints]
        self.yTrain = self.y[:self.nrRefPoints]
        self.zTrain = self.z[:self.nrRefPoints]
        self.uTrain = self.u[:self.nrRefPoints]
        self.vTrain = self.v[:self.nrRefPoints]

        self.xTest = self.x[self.nrRefPoints:]
        self.yTest = self.y[self.nrRefPoints:]
        self.zTest = self.z[self.nrRefPoints:]
        self.uTest = self.u[self.nrRefPoints:]
        self.vTest = self.v[self.nrRefPoints:]


    def createMatrix(self, u_or_v):
        """
        Helper function to create the Au or Av matrix for the DLT system.
        """
        ones_col = np.ones(len(self.xTrain))
        zeros_cols = np.zeros((len(self.xTrain), 4))

        if u_or_v == 'u':
            u = self.uTrain
            factor = -u
            matrix = np.column_stack([
                self.xTrain,
                self.yTrain,
                self.zTrain,
                ones_col,
                zeros_cols,
                factor * self.xTrain,
                factor * self.yTrain,
                factor * self.zTrain
            ])
        elif u_or_v == 'v':
            v = self.vTrain
            factor = -v
            matrix = np.column_stack([
                zeros_cols,
                self.xTrain,
                self.yTrain,
                self.zTrain,
                ones_col,
                factor * self.xTrain,
                factor * self.yTrain,
                factor * self.zTrain
            ])
        else:
            raise ValueError("u_or_v must be either 'u' or 'v'")

        return matrix


    def constructA(self):
        """
        Constructs the matrix A by concatenating Au and Av.
        """
        Au = self.createMatrix('u')
        Av = self.createMatrix('v')
        A = np.vstack([Au, Av])
        return A


    def constructU(self):
        """
        Constructs the vector U by concatenating uTrain and vTrain.
        """
        U = np.hstack([self.uTrain, self.vTrain]).reshape(-1, 1)
        return U


    def computeL(self):
        """
        Computes the parameter vector L using least squares solution.
        """
        A = self.constructA()
        U = self.constructU()
        L = np.linalg.lstsq(A, U, rcond=None)[0]
        return L


    def compute_uv(self, X, L, data_type='train'):
        """
        Helper function to compute u and v values for a given dataset (train or test).
        Selects the correct data (train or test) based on `data_type`.
        """
        if data_type == 'train':
            X_data = np.column_stack([self.xTrain, self.yTrain, self.zTrain])
        elif data_type == 'test':
            X_data = np.column_stack([self.xTest, self.yTest, self.zTest])
        else:
            raise ValueError("data_type must be either 'train' or 'test'")

        X_h = np.column_stack([X_data, np.ones(len(X_data))])
        denominator = X_data @ L[8:11] + 1
        u = (X_h @ L[:4]) / denominator
        v = (X_h @ L[4:8]) / denominator
        return u, v

    def computeIntrinsics(self, L):
        """
        Compute the intrinsic camera parameters from the DLT parameter vector.

        Parameters:
        -----------
        L : array-like
            The 11-element DLT parameter vector (flattened).

        Returns:
        --------
        dict
            A dictionary containing the intrinsic parameters:
            - 'Lconst': Scaling constant
            - 'up': Principal point in u direction (u0)
            - 'vp': Principal point in v direction (v0)
            - 'bx': Camera constant in x (bx = b / k_u)
            - 'by': Camera constant in y (by = b / k_v)
        """
        L = np.asarray(L).flatten()
        Lconst = -1 / np.sqrt(np.sum(L[8:11] ** 2))

        # Principal point
        up = Lconst ** 2 * np.sum(L[0:3] * L[8:11])
        vp = Lconst ** 2 * np.sum(L[4:7] * L[8:11])

        # Camera constants
        bx = np.sqrt(Lconst ** 2 * np.sum(L[0:3] ** 2) - up ** 2)
        by = np.sqrt(Lconst ** 2 * np.sum(L[4:7] ** 2) - vp ** 2)

        print(f"Principal point (u0, v0): ({up:.2f}, {vp:.2f})")
        print(f"Camera constants bx: {bx:.2f}, by: {by:.2f}")

        return {
            'Lconst': Lconst,
            'up': up,
            'vp': vp,
            'bx': bx,
            'by': by
        }

    def computeExtrinsics(self, L, intrinsics):
        """
        Compute the extrinsic camera parameters (rotation matrix and translation vector).

        Parameters:
        -----------
        L : array-like
            The 11-element DLT parameter vector (flattened).
        intrinsics : dict
            Dictionary containing the intrinsic parameters returned by compute_intrinsics().

        Returns:
        --------
        dict
            A dictionary containing the extrinsic parameters:
            - 'R': 3x3 rotation matrix
            - 'T': 3x1 translation vector (optical center in world coordinates)
        """
        L = np.asarray(L).flatten()
        Lconst = intrinsics['Lconst']
        up = intrinsics['up']
        vp = intrinsics['vp']
        bx = intrinsics['bx']
        by = intrinsics['by']

        # Rotation matrix
        r11 = Lconst * (up * L[8] - L[0]) / bx
        r12 = Lconst * (vp * L[8] - L[4]) / by
        r13 = Lconst * L[8]
        r21 = Lconst * (up * L[9] - L[1]) / bx
        r22 = Lconst * (vp * L[9] - L[5]) / by
        r23 = Lconst * L[9]
        r31 = Lconst * (up * L[10] - L[2]) / bx
        r32 = Lconst * (vp * L[10] - L[6]) / by
        r33 = Lconst * L[10]

        R = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        # Optical center (translation vector)
        A = np.vstack([L[0:3], L[4:7], L[8:11]])
        b = np.array([L[3], L[7], 1])
        T = -np.linalg.solve(A, b)

        print("Rotation Matrix R:")
        print(R)
        print("Optical Center (Translation Vector):")
        print(f"x: {T[0]:.2f}, y: {T[1]:.2f}, z: {T[2]:.2f}")

        return {
            'R': R,
            'T': T
        }

    def predict(self):
        """
        Compute the predicted u and v values for both training and testing data.
        """
        L = self.computeL().flatten()
        Xtrain = np.column_stack([self.xTrain, self.yTrain, self.zTrain])
        uEst, vEst = self.compute_uv(Xtrain, L, data_type='train')
        Xtest = np.column_stack([self.xTest, self.yTest, self.zTest])
        uVal, vVal = self.compute_uv(Xtest, L, data_type='test')
        self.uPred = np.concatenate((uEst, uVal))
        self.vPred = np.concatenate((vEst, vVal))

        # Intrinsic parameters
        intrinsics = self.computeIntrinsics(L)

        # Extrinsic parameters
        extrinsics = self.computeExtrinsics(L, intrinsics)


    def visualiseResults(self, img):
        """
        Function to visualize the training and testing points and the estimated points on an image.
        """
        plt.imshow(img)
        plt.scatter(self.uTrain, self.vTrain, marker='x', color='r', label=f'Calibration Points ({self.nrRefPoints})')
        plt.scatter(self.uTest, self.vTest, marker='x', color='#00FF00', label=f'Test Points ({len(self.x) - self.nrRefPoints})')
        plt.scatter(self.uPred, self.vPred, marker='o', edgecolor='orange', facecolors='none', s=50, label='Estimated Points')
        plt.legend()
        plt.title(f'Direct Linear Transformation')
        plt.show()


def main():
    # Read the image
    img = cv2.imread("dlt.png")

    # Check if image was read successfully
    if img is None:
        print("Error: Image not found or failed to load.")
        return

    nrRefPoints = 6
    d = 30e-3  # scale factor for the world coordinates
    x = -d * np.array([0, 1, 4, 2, 3, 5, 3, 4, 0, 2, 0, 2, 1, 5, 2])
    y = d * np.array([0, 2, 1, 4, 4, 3, 2, 4, 4, 0, 3, 2, 4, 0, 4])
    z = d * np.array([0, 0, 0, 1, 3, 0, 0, 2, 3, 0, 0, 0, 2, 0, 0])

    u = np.array([371, 377, 185, 377, 310, 210, 271, 270, 476, 248, 465, 320, 424, 99, 380])
    v = np.array([405, 326, 326, 201, 84, 263, 307, 139, 89, 377, 302, 314, 148, 340, 254])

    # Initialize the DLT class with the provided coordinates and reference points
    dlt = DLT(x, y, z, u, v, nrRefPoints)

    # Perform predictions
    dlt.predict()

    # Visualize the results
    dlt.visualiseResults(img)


if __name__ == "__main__":
    main()