from opendr.contexts.ctx_mesa import OsContext
from opendr.contexts._constants import *
import numpy as np

class Renderer:
    def __init__(self, in_camera, in_imSize):
        width = in_imSize[0]
        height = in_imSize[1]
        self.gl = OsContext(width, height, typ = GL_FLOAT)
        # set camera
        self.gl.MatrixMode(GL_PROJECTION)
        self.gl.LoadIdentity();

        pixel_center_offset = 0.5
        f = np.mean(in_camera.focalLength)
        cx = in_camera.principalPoint[0]
        cy = in_camera.principalPoint[1]
        near = 1.
        far = 1e2
        right  =  (width  - (cx + pixel_center_offset)) * (near / f)
        left   =          - (cx + pixel_center_offset)  * (near / f)
        top    = -(height - (cy + pixel_center_offset)) * (near / f)
        bottom =            (cy + pixel_center_offset)  * (near / f)
        self.gl.Frustum(left, right, bottom, top, near, far)

        self.gl.MatrixMode(GL_MODELVIEW);
        self.gl.LoadIdentity(); # I
        self.gl.Rotatef(180, 1, 0, 0) # I * xR(pi)

        view_matrix = in_camera.GetViewMatrix()

        view_mtx = np.asarray(np.vstack((view_matrix, np.array([0, 0, 0, 1]))), np.float32, order='F')
        self.gl.MultMatrixf(view_mtx) # I * xR(pi) * V

        self.gl.Enable(GL_DEPTH_TEST)
        self.gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.gl.Disable(GL_LIGHTING)
        self.gl.Disable(GL_CULL_FACE)
        self.gl.PixelStorei(GL_PACK_ALIGNMENT,1)
        self.gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)

        self.gl.UseProgram(0)

    def Render(self, in_vertexArray, in_faceArray, in_colorArray = None):
        self.gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        self.gl.EnableClientState(GL_VERTEX_ARRAY)
        if (in_colorArray is None):
            self.gl.DisableClientState(GL_COLOR_ARRAY)
        else:
            self.gl.EnableClientState(GL_COLOR_ARRAY)
        self.gl.VertexPointer(np.ascontiguousarray(in_vertexArray).reshape((-1,3)))
        if not (in_colorArray is None):
            self.gl.ColorPointerd(np.ascontiguousarray(in_colorArray).reshape((-1, 3)))
        self.gl.DrawElements(GL_TRIANGLES, np.asarray(in_faceArray, np.uint32).ravel())
        return self.gl.getImage()
