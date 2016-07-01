import numpy as np
from opendr.contexts.ctx_mesa import OsContext
from opendr.contexts._constants import *
import transforms3d.euler as tf

import sys
sys.path.insert(0, "lib")
from shapemodel import shapepose

sys.path.insert(0, "sample")
from readModel import ReadModel
import camera
from renderer import Renderer
from projection import *

'''
Returns normalized array of person vertices on the ground plane (z = 0) at the point (0, 0)
* io_vertexArray is the input array of vertices
* io_jointArray  is the input array of joints
* in_sizeRatio   is the ratio between the result model size and the input model size
'''
def normalizeVertices(io_vertexArray, io_jointArray, in_personModel):
    minZ = np.min(io_vertexArray[:, 2])
    io_vertexArray[:, :2] -= in_personModel['meanLocation'][:, :2]
    io_vertexArray[:, 2] -= minZ
    io_vertexArray *= in_personModel['sizeRatio']
    io_jointArray[:, 4:6] -= in_personModel['meanLocation'][:, :2]
    io_jointArray[:, 6] -= minZ
    io_jointArray[:, 1 : 7] *= in_personModel['sizeRatio']


'''
Returns person model with a standard shape in a standard pose at (0, 0)
* out_jointArray  is the output array of joints
* out_vertexArray is the output array of vertices
* out_faceArray   is the output array of mesh faces
* out_colorArray  is the output array of the vertex colors
'''
def getBaseMesh():
  personModel = ReadModel('caesar-norm-nh', 1.75)
  # parameters of the human pose
  poseParams = np.zeros((1, 31), dtype = np.float64)
  # parameters of the human shape
  shapeParams = np.zeros((1, personModel["eValues"].size))

  out_vertexArray, out_jointArray = shapepose(poseParams, shapeParams,
      personModel['eVectors'], personModel['modelPath'])
  normalizeVertices(out_vertexArray, out_jointArray, personModel)

  out_colorArray = np.ones((out_vertexArray.shape[0], 3), dtype = np.float64)
  out_facesArray = personModel["faceArray"].copy()

  mask = out_vertexArray[:, 2] >= out_jointArray[11, 6]
  for x in [0, 2]:
    out_colorArray[mask, x] = 0

  return out_jointArray, out_vertexArray, out_facesArray, out_colorArray

pitch = 1.92
roll = 0
height = 7.5

focalLength = 2700
imgWidth = 1920
imgHeight = 1080

cam = camera.Camera()
# camera location in the scene (it lies on the Z axis)
cam.SetLocation(np.array([0, 0, height]))
# camera view direction
R = tf.euler2mat(0, pitch, roll, "szxz")
cam.SetRotationMatrix(R)
# focal length
cam.SetFocalLength(np.array([focalLength, focalLength]))
# principal point in the image center
cam.SetPrincipalPoint(np.array([(imgWidth - 1) / 2., (imgHeight - 1) / 2.]))
cam.SetSkew(0)

P = cam.GetProjectionMatrix()
imgCenter = np.array([[imgWidth / 2], [imgHeight / 2]])
groundPlane = np.array([0, 0, 1, 0]) # z = 0
groundLocation = Hom2Het(Im2World(P, imgCenter, groundPlane))

jointArray, vertexArray, facesArray, colorArray = getBaseMesh()
# place person in a center of the image
for idx in range(2):
  vertexArray[:, idx] += groundLocation[idx]

rn = Renderer(cam, [imgWidth, imgHeight])
img = rn.Render(vertexArray, facesArray, colorArray)

import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()
