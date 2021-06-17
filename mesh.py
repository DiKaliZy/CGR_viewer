from OpenGL.GL import *
import numpy as np

# mesh 미리 그려서 glDict에 저장
def GLCreateList(glDict, newName, drawFunction):
    newID = glGenLists(1)
    glNewList(newID, GL_COMPILE)
    drawFunction()
    glEndList()
    glDict[newName] = newID

def drawPlate():
    glBegin(GL_QUADS)
    glNormal3fv(np.array([0., 1., 0.]))
    glVertex3fv(np.array([0. , 0., 0.]))
    glVertex3fv(np.array([0. , 0., 1.]))
    glVertex3fv(np.array([1. , 0., 1.]))
    glVertex3fv(np.array([1. , 0., 0.]))

    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(1, -0.001, 0)
    glVertex3f(1, -0.001, 1)
    glVertex3f(0, -0.001, 1)
    glVertex3f(0, -0.001, 0)

    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(1, -0.001, 1)
    glVertex3f(0, -0.001, 1)
    glVertex3f(0, 0, 1)
    glVertex3f(1, 0, 1)

    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f(1, 0, 1)
    glVertex3f(1, -0.001, 1)
    glVertex3f(1, -0.001, 0)
    glVertex3f(1, 0, 0)

    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(1, 0, 0)
    glVertex3f(1, -0.001, 0)
    glVertex3f(0, -0.001, 0)
    glVertex3f(0, 0, 0)

    glNormal3f(-1.0, 0.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, -0.001, 0)
    glVertex3f(0, -0.001, 1)
    glVertex3f(0, 0, 1)

    glEnd()

def draw_mesh(type, start= (), end=None, glDict = None, size= 1.0,
              x_size = None, y_size = None, z_size = None):
    if type == "LINE":
        glLineWidth(size)
        glBegin(GL_LINES)
        glVertex3fv(start)
        glVertex3fv(end)
        glEnd()
        glLineWidth(1.0)
    elif type == "PLATE":
        glCallList(glDict[type])

    elif type == "BOX":
        parent2child = end - start
        if np.sqrt((parent2child) @ np.transpose(parent2child)) != 0:
            newy = (parent2child) / np.sqrt((parent2child) @ np.transpose(parent2child))
        else:
            newy = np.array([0, 1, 0])

        if newy[0] == 0. and (newy[1] == 1. or newy[1] == -1.) and newy[2] == 0.:
            rotmat = np.identity(3)
        else:
            rotaxis = np.cross(newy, np.array([0., 1., 0.]))
            if np.sqrt(rotaxis @ np.transpose(rotaxis)) == 0:
                print(newy)
            rotaxis = rotaxis / np.sqrt(rotaxis @ np.transpose(rotaxis))
            newz = np.cross(rotaxis, newy)
            newz = newz / np.sqrt(newz @ np.transpose(newz))
            rotmat = np.column_stack((rotaxis, newy, newz))

        glPushMatrix()

        glTranslatef(parent2child[0] / 2, parent2child[1] / 2, parent2child[2] / 2)

        mat = np.identity(4)
        mat[:3, :3] = rotmat
        glMultMatrixf(mat.T)

        glPushMatrix()
        if len(start)>0:
            glTranslatef(start[0],start[1],start[2])

        length = np.sqrt((end - start) @
                         np.transpose(end - start))
        glScale(size / 100, length, size / 100)

        glCallList(glDict[type])
        glPopMatrix()
        glPopMatrix()

    elif type == "SPHERE":
        glPushMatrix()
        if len(start) > 0:
            glTranslatef(start[0], start[1], start[2])
        glScale(size / 10, size / 10, size / 10)

        glCallList(glDict[type])
        glPopMatrix()

def drawBox():
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)

    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)

    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    glNormal3f(-1.0, 0.0, 0.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)

    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, -0.5)

    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)

    glEnd()

def drawSphere(numLats=10, numLongs=5):
    for i in range(0, numLats + 1):
        lat0 = np.pi * (-0.5 + float(float(i-1)/float(numLats)))
        z0 = np.sin(lat0)
        zr0 = np.cos(lat0)

        lat1 = np.pi * (-0.5 + float(float(i) / float(numLats)))
        z1 = np.sin(lat1)
        zr1 = np.cos(lat1)

        glBegin(GL_QUAD_STRIP)

        for j in range(0, numLongs + 1):
            lng = 2*np.pi * float(float(j-1)/float(numLongs))
            x = np.cos(lng)
            y = np.sin(lng)
            glVertex3f(x*zr0,y*zr0,z0)
            glVertex3f(x*zr1,y*zr1,z1)

        glEnd()