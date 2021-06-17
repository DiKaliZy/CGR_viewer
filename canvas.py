import wx
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math
from wx.glcanvas import *

import mesh
import utility
import kinematics as IK
import physics


class GLCanvasBase(glcanvas.GLCanvas):
    def __init__(self, parent, messenger):

        # linux ubuntu 환경에서 DEPTH_TEST 제대로 작동시키기 위해 필요한 부분.
        # 왜 이렇게 해야 작동하는지는 모르겠음.
        # 참조 주소 :
        # https://discuss.wxpython.org/t/opengl-depth-buffer-deosnt-seem-to-work-in-wx-glcanvas/27513/10
        attribs = [WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 24]
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs)

        self.frame = parent.GetParent()
        self.messenger = messenger
        self.init = False
        self.shift = False
        self.clicked = False
        self.at = np.array([0, 0, 0])
        self.up = np.array([0, 1, 0])
        self.leng = 10
        self.azimuth = np.radians(30)
        self.elevation = np.radians(30)
        self.cam = np.array(
            [self.leng * np.sin(self.azimuth), self.leng * np.sin(self.elevation), self.leng * np.cos(self.azimuth)])

        self.model_list = []
        self.timeset = (1 / 30) * 1000  # ms단위
        self.isrepeat = True
        self.timer = wx.Timer(self)

        self.skeleton_view = False  # line으로 골격만 표현
        self.frame_view = False  # mesh, frame 설정
        self.velocity_view = False  # velocity line 표현 설정
        self.joint_view = False  # joint position visualization 설정
        self.time_warping_view = False
        self.motion_warping_view = False
        self.stitching_view = False
        self.blending_view = False
        self.original_view = True

        self.selected_character_id = 0  # 캐릭터 클릭
        self.selected_joint_id = 0  # 관절 클릭

        self.playSpeed = 2  # 0 = 0.25배, 1 = 0.5배, 2= 1배

        self.context = glcanvas.GLContext(self)

        self.timer.Start(self.timeset)

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnMouseDown)  # camera rotate
        self.Bind(wx.EVT_RIGHT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)  # shift + mouse drag -> camera move
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)
        self.Bind(wx.EVT_MOUSEWHEEL, self.Wheel)  # zoom in, zoom out
        self.Bind(wx.EVT_TIMER, self.OnTime)

    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)

    def OnTime(self, event):
        self.Refresh()

    # 마우스 클릭 조작
    def OnMouseDown(self, event):
        self.CaptureMouse()
        self.mx, self.my = self.lastmx, self.lastmy = event.GetPosition()
        if self.clicked == False:
            self.clicked = True
        else:
            event.Skip()

    def OnMouseUp(self, event):
        self.ReleaseMouse()
        self.clicked = False

    # 마우스 움직임 조작
    def OnMouseMotion(self, event):
        if self.clicked == True:
            self.lastmx, self.lastmy = self.mx, self.my
            self.mx, self.my = event.GetPosition()
            normy = (self.my - self.lastmy) * np.sqrt(np.dot(self.cam - self.at, self.cam - self.at)) / 120
            normx = -(self.mx - self.lastmx) * np.sqrt(np.dot(self.cam - self.at, self.cam - self.at)) / 120
            normry = (self.my - self.lastmy) / 100
            normrx = -(self.mx - self.lastmx) / 100
            # Camera Panning
            if self.shift == True:
                self.getWUV()
                paramU = self.u * normx
                paramV = self.v * normy
                param = paramU + paramV
                self.cam = self.cam + param
                self.at = self.at + param

                self.Refresh(False)

            # Camera Rotate(Orbit)
            elif self.shift == False:
                self.leng = np.sqrt(np.dot(self.cam - self.at, self.cam - self.at))

                # Elevation
                if self.elevation + normry <= np.radians(90) and self.elevation + normry >= np.radians(-90):
                    self.elevation += normry
                elif self.elevation + normry >= np.radians(90):
                    self.elevation = np.radians(89.9)
                elif self.elevation + normry <= np.radians(-90):
                    self.elevation = np.radians(-89.9)
                # Azirmuth
                self.azimuth += normrx
                self.cam = np.array([self.leng * np.cos(self.elevation) * np.sin(self.azimuth),
                                     self.leng * np.sin(self.elevation),
                                     self.leng * np.cos(self.elevation) * np.cos(self.azimuth)])
                self.cam += self.at
                self.Refresh(False)
        else:
            event.Skip()

    # 키 입력
    def OnKeyDown(self, event):
        move = 2  # 키보드 입력으로 이동시킬 frame 수
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_SHIFT:
            if self.shift == False:
                self.shift = True
            else:
                event.Skip()
        elif keycode == wx.WXK_SPACE:
            state = self.messenger.change_play_state()
            self.messenger.change_play_button(state)
        elif keycode == wx.WXK_LEFT:
            if self.shift:
                self.messenger.change_frame(-move, is_relative_value=True, allow_pinned=True)
            else:
                self.messenger.change_frame(-move, is_relative_value=True)
            self.Refresh()
        elif keycode == wx.WXK_RIGHT:
            if self.shift:
                self.messenger.change_frame(move, is_relative_value=True, allow_pinned=True)
            else:
                self.messenger.change_frame(move, is_relative_value=True)
            self.Refresh()
        elif keycode == wx.WXK_UP:
            ...
        elif keycode == wx.WXK_DOWN:
            ...
        # mesh/skeleton 전환
        elif keycode == ord('Z'):
            self.skeleton_view = self.change_state(self.skeleton_view)
        # Z키 입력 - polygon / frame 전환
        elif keycode == ord('X'):
            self.frame_view = self.change_state(self.frame_view)
        elif keycode == ord('0'):
            if self.original_view:
                self.original_view = False
            else:
                self.original_view = True
        # 1 입력 - joint view toggle
        elif keycode == ord('1'):
            self.joint_view = self.change_state(self.joint_view)
            self.messenger.change_joint_button(self.joint_view)
        # 2 입력 - velocity line view toggle
        elif keycode == ord('2'):
            self.velocity_view = self.change_state(self.velocity_view)
            self.messenger.change_velocity_button(self.velocity_view)
        elif keycode == ord('3'):
            if self.time_warping_view:
                self.time_warping_view = False
            else:
                self.time_warping_view = True
                self.messenger.time_warp_character(self.selected_character_id, self.time_warp_func)
        elif keycode == ord('4'):
            if self.motion_warping_view:
                self.motion_warping_view = False
            else:
                self.motion_warping_view = True
                self.messenger.motion_warp_character(self.selected_character_id, self.keyframe_list,
                                                     self.start_delta, self.end_delta, self.motion_warp_func)
        elif keycode == ord('5'):
            if self.stitching_view:
                self.stitching_view = False
            else:
                self.stitching_view = True
                self.messenger.motion_stitch_character(self.selected_character_id, self.clip1_start, self.clip1_size,
                                                       self.selected_character2_id, self.clip2_start, self.clip2_size,
                                                       self.transition_length, self.stitch_func)
        elif keycode == ord('6'):
            if self.blending_view:
                self.blending_view = False
            else:
                self.blending_view = True
                self.messenger.motion_blend_character(self.selected_character_id, self.selected_character2_id,
                                                      self.motion1_seg_start, self.motion2_seg_start,
                                                      self.motion1_seg_size, self.motion2_seg_size,
                                                      self.blending_length, self.blending_func)
        # u - x 방향 +1
        elif keycode == ord('U'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, x_mod=-0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        # j - x 방향 -1
        elif keycode == ord('J'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, x_mod=0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        # i - y 방향 +1
        elif keycode == ord('I'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, y_mod=-0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        # k - y 방향 -1
        elif keycode == ord('K'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, y_mod=0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        # o - z 방향 +1
        elif keycode == ord('O'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, z_mod=-0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        # l - z 방향 -1
        elif keycode == ord('L'):
            self.messenger.call_IK_mod(self.selected_character_id, self.selected_joint_id, z_mod=0.01)
            keyframes = self.messenger.get_IK_keyframes(self.selected_character_id)
            self.keyframe_list = keyframes
        else:
            event.Skip()

    def OnKeyUp(self, event):
        if event.GetKeyCode() == wx.WXK_SHIFT:
            self.shift = False
        else:
            event.Skip()

    # 마우스 휠 -> 줌인, 줌아웃
    def Wheel(self, event):
        self.getWUV()
        paramW = self.w * 5
        # wheel up -> zoom in
        if event.GetWheelRotation() > 0:
            if np.sqrt(np.dot(self.cam - paramW - self.at, self.cam - paramW - self.at)) >= np.sqrt(
                    np.dot(paramW, paramW)):
                self.cam = self.cam - paramW
                self.Refresh(False)
            elif np.sqrt(np.dot(self.cam - self.w - self.at, self.cam - self.w - self.at)) >= 2 * np.sqrt(
                    np.dot(self.w, self.w)):
                self.cam = self.cam - self.w
                self.Refresh(False)
            else:
                event.Skip()
        # wheel down -> zoom out
        elif event.GetWheelRotation() < 0:
            self.cam = self.cam + paramW
            self.Refresh(False)
        else:
            event.Skip()

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW.

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

    # 카메라 좌표 계산
    def getWUV(self):
        self.w = (self.cam - self.at) / np.sqrt(np.dot((self.cam - self.at), (self.cam - self.at)))
        if np.dot(np.cross(self.up, self.w), np.cross(self.up, self.w)) != 0:
            self.u = np.cross(self.up, self.w) / np.sqrt(np.dot(np.cross(self.up, self.w), np.cross(self.up, self.w)))
        else:
            self.u = np.array([np.sin(self.azimuth), 0, np.cos(self.azimuth)])
        self.v = np.cross(self.w, self.u) / np.sqrt(np.dot(np.cross(self.w, self.u), np.cross(self.w, self.u)))

    def change_state(self, target):
        if target:
            target = False
        else:
            target = True
        return target

    def change_property_view_state(self, target='joint_view'):
        state = False
        if target == 'joint_view':
            self.joint_view = self.change_state(self.joint_view)
            state = self.joint_view
        elif target == 'velocity_view':
            self.velocity_view = self.change_state(self.velocity_view)
            state = self.velocity_view
        return state


class Canvas(GLCanvasBase):
    def InitGL(self):

        self.light_pos = [0., 50., 20., 1.]
        self.glDict = {}

        # TODO: 임시로 IK 대상 설정
        self.selected_character_id = 0
        self.selected_character2_id = 1
        self.selected_joint_id = 6

        # TODO: 임시로 time warping 함수 설정
        def time_warp_scale_func(t):
            s = 0.5 * t
            return s
        self.time_warp_func = time_warp_scale_func

        # TODO: 임시로 motion warping 함수 설정
        def motion_warp_func(t):
            s = math.sin(t * (math.pi/2))
            return s
        self.motion_warp_func = motion_warp_func
        self.keyframe_list = []
        self.start_delta = 10
        self.end_delta = 10

        # TODO: 임시로 stitching 함수 설정
        def stitch_func(t):
            s = math.sin(t * (math.pi/2))
            return s
        self.stitch_func = stitch_func
        self.transition_length = 10
        self.clip1_start = 0
        self.clip2_start = 60 # 10  # 60
        self.clip1_size = 80 # 40   # 80
        self.clip2_size = 200


        # TODO: 임시로 blending 함수 설정
        def blending_func(t):
            s = math.sin(t*(math.pi/2))
            return s
        self.motion1_seg_start = 61
        self.motion1_seg_size = 20
        self.motion2_seg_start = 61
        self.motion2_seg_size = 20
        self.blending_length = 20
        self.blending_func = blending_func


        # TODO: 임시로 particle 객체 생성
        self.particle = self.messenger.make_test_particle(np.array([0., 2., 0.]), np.array([1., 0., 0.]), 1.)
        # self.particle2 = self.messenger.make_test_particle(np.array([0., 1., 0.]), np.array([0., 0., 0.]), 1.)

        # self.messenger.make_test_spring(self.particle, self.particle2)
        self.time_step = 0.003


        mesh.GLCreateList(self.glDict, 'SPHERE', mesh.drawSphere)
        mesh.GLCreateList(self.glDict, 'BOX', mesh.drawBox)
        mesh.GLCreateList(self.glDict, 'PLATE', mesh.drawPlate)

        glShadeModel(GL_SMOOTH)
        glClearColor(0.96, 0.96, 0.9, 1.)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(45, 1., 1., 1000)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glClearDepth(1.)

    def OnDraw(self):
        gridlane = 10
        gridscale = 1
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)

        '''
        glColor3f(1.0, 1.0, 1.0)
        timestep = str(self.time_step)
        for ch in timestep:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(ch)))
        '''
        glLoadIdentity()

        # gluLookAt(self.light_pos[0], self.light_pos[1], self.light_pos[2], self.at[0], self.at[1], self.at[2], 0, 1, 0)
        gluLookAt(self.cam[0], self.cam[1], self.cam[2], self.at[0], self.at[1], self.at[2], 0, 1, 0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # draw checker_box
        for i in range(-2 * gridlane, 2 * gridlane + 1):
            for j in range(-2 * gridlane, 2 * gridlane + 1):
                glPushMatrix()
                glScale(gridscale, gridscale, gridscale)
                if (j + i) % 2 == 0:
                    glColor3f(1., 1., 1.)
                    glTranslatef(i, 0., j)
                    mesh.draw_mesh("PLATE", glDict=self.glDict)
                else:
                    glColor3f(.8, .8, .8)
                    glTranslatef(i, 0., j)
                    mesh.draw_mesh("PLATE", glDict=self.glDict)
                glPopMatrix()

        # draw grid
        glColor3f(0.3, 0.3, 0.3)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBegin(GL_LINES)
        for i in range(-2 * gridlane, gridlane * 2 + 2):
            glVertex3fv(np.array([i * gridscale, 0.001, gridlane * gridscale * 2 + 1]))
            glVertex3fv(np.array([i * gridscale, 0.001, gridlane * -gridscale * 2]))
            glVertex3fv(np.array([gridlane * gridscale * 2 + 1, 0.001, gridscale * i]))
            glVertex3fv(np.array([gridlane * -gridscale * 2, 0.001, gridscale * i]))
        glEnd()

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_RESCALE_NORMAL)
        glEnable(GL_NORMALIZE)

        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.])
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_pos)
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1., 1., 1., 1.])

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1., 1., 1., 1.])
        glMaterialfv(GL_FRONT, GL_SHININESS, 10)

        if self.frame_view == False:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        self.draw_model()
        glDisable(GL_LIGHTING)

        self.SwapBuffers()

    # model별 그림자
    def draw_shadow(self, model, light_pos):
        for joint in model.joint:
            start = np.array(
                [joint.motion_pos[model.motion.frame][0], joint.motion_pos[model.motion.frame][1],
                 joint.motion_pos[model.motion.frame][2]])
            for child in joint.child:
                end = np.array([child.motion_pos[model.motion.frame][0],
                                child.motion_pos[model.motion.frame][1],
                                child.motion_pos[model.motion.frame][2]])

                # 각 관절 별 projection 좌표 구하기
                projection_mat = np.identity(4)
                projection_mat[:3, :4] = utility.spot_light_projection_mat(light_pos, start)
                shadow_start = projection_mat @ np.append(start, np.array([1.]))
                start = shadow_start[:3]
                projection_mat = np.identity(4)
                projection_mat[:3, :4] = utility.spot_light_projection_mat(light_pos, end)
                shadow_end = projection_mat @ np.append(end, np.array([1.]))
                end = shadow_end[:3]

                if self.skeleton_view == True:
                    mesh.draw_mesh("LINE", start=start, end=end)
                else:
                    parent2child = end - start
                    glPushMatrix()

                    glTranslatef(start[0], 0.0001 - parent2child[1] / 2, start[2])

                    mesh.draw_mesh("BOX",
                                   start=start,
                                   end=end,
                                   glDict=self.glDict, size=joint.scale)
                    glPopMatrix()

    # 모델 그리기
    def draw_model(self):
        character_list = self.messenger.get_character_id_list()
        for character_id in character_list:
            glPushMatrix()
            character_state_dict, _, _, _ = self.get_character_info(character_id)
            now_frame = character_state_dict["now_frame"]
            moded_type = character_state_dict["moded_type"]

            IK_limb_check, target_positions = self.messenger.check_IK(character_id, now_frame)
            if IK_limb_check:
                _, rotmats, positions, velocities = self.get_character_info(character_id, category="limb")
                self.draw_limb_IK(character_state_dict, rotmats, positions)
                if self.joint_view == True:
                    self.draw_joint(positions)
                glColor3f(0.9, 0.3, 0.3)
                self.draw_IK_target_point(target_positions)
            IK_jacobian_check, target_positions = self.messenger.check_IK(character_id, now_frame, category="jacobian")
            if IK_jacobian_check:
                _, rotmats, positions, velocities = self.get_character_info(character_id, category="jacobian")
                self.draw_jacob_IK(character_state_dict, rotmats, positions)
                if self.joint_view == True:
                    self.draw_joint(positions)
                glColor3f(0.1, 0.1, 0.1)
                self.draw_IK_target_point(target_positions)

            if self.time_warping_view and moded_type == "time_warping":
                # frame당 모션 그리기
                _, rotmats, positions, velocities = self.get_character_info(character_id)
                self.draw_character(character_state_dict, rotmats, positions, model_color=(0.8, 0.1, 0.1))
                # joint 그리기
                if self.joint_view == True:
                    self.draw_joint(positions)

            if self.motion_warping_view:
                if moded_type == "motion_warping":
                    # frame당 모션 그리기
                    _, rotmats, positions, velocities = self.get_character_info(character_id)
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.5, 0.5, 0.1))
                    # joint 그리기
                    if self.joint_view == True:
                        self.draw_joint(positions)
                keyframes = self.messenger.get_IK_keyframes(character_id)
                for keyframe in keyframes:
                    character_state_dict, rotmats, positions, velocities =\
                        self.get_character_info(character_id, now_frame=keyframe, category="limb")
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.1, 0.8, 0.8))

            if self.stitching_view:
                if moded_type == "motion_stitching":
                    # frame당 모션 그리기
                    _, rotmats, positions, velocities = self.get_character_info(character_id)
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.3, 0.8, 0.3))
                    # joint 그리기
                    if self.joint_view == True:
                        self.draw_joint(positions)
                if moded_type == "motion_clipping":
                    # frame당 모션 그리기
                    _, rotmats, positions, velocities = self.get_character_info(character_id)
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.5, 0.5, 0.1))
                    # joint 그리기
                    if self.joint_view == True:
                        self.draw_joint(positions)
                '''
                if moded_type == "motion_align":
                    # frame당 모션 그리기
                    _, rotmats, positions, velocities = self.get_character_info(character_id)
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.5, 0.5, 0.1))
                    # joint 그리기
                    if self.joint_view == True:
                        self.draw_joint(positions)
                '''
            if self.blending_view:
                if moded_type == "motion_blending":
                    # frame당 모션 그리기
                    _, rotmats, positions, velocities = self.get_character_info(character_id)
                    self.draw_character(character_state_dict, rotmats, positions, model_color=(0.5, 0.5, 0.1))
                    # joint 그리기
                    if self.joint_view == True:
                        self.draw_joint(positions)

            if self.original_view and moded_type == "original":
                # frame당 모션 그리기
                _, rotmats, positions, velocities = self.get_character_info(character_id)
                self.draw_character(character_state_dict, rotmats, positions)
                # joint 그리기
                if self.joint_view == True:
                    self.draw_joint(positions)
                # 속도 시각화
                if self.velocity_view == True:
                    self.draw_velocity(positions, velocities)

            glPopMatrix()

            glPushMatrix()
            # 그림자 그리기 위해 lighting 제거
            glDisable(GL_LIGHTING)

            glColor3f(0.2, 0.2, 0.2)
            # ====================================================================================
            # 현재 구현은 단순히 그림자를 xz평면에 각 관절 좌표값을 투영하여 기존 model을 그리듯이 xz 평면 상에 그린 것
            # 최소 목표 : 광원에 따른 projection matrix를 구해서 draw_shadow 호출 할 필요 없이
            #           glMultMatrix 후 draw model 하면 바로 그림자 그려지도록 하기
            # 가능하면 shadow map 같은 거 써서 구현할 수 있도록 해 보기?
            # ====================================================================================
            # 1. spot light 환경에서 각 관절 별 projection 좌표 구해서 hierarchical하게 하나하나 그리는 방식
            # (spot light : 광원에서 나온 빛이 방사형으로 쏘아져 각 부분별로 다르게 projection)
            # self.draw_shadow(model, self.light_pos)
            # =====================================================================================
            # 2. direction light 환경에서 light source 좌표와 root 좌표가 이루는 vector를 light direction으로 정해서 projection
            # (direction light : 이론상 광원이 무한대 떨어진 거리에 위치하여 모든 vertex가 동일한 direction의 빛을 받는다 가정)
            ## projection = np.identity(4)
            ## projection[:3, :4] = utility.direction_light_projection_mat(self.light_pos,
            ##                                                            root.motion_pos[model.frame])
            ## glMultMatrixf(projection.T)
            ## self.draw_motion(model, shadow= True)
            # self.draw_joint(model, type="root")
            # =====================================================================================
            # 3. spot light projection matrix 사용 (TODO)
            # =====================================================================================
            glPopMatrix()
            glEnable(GL_LIGHTING)
        if len(character_list) == 0:
            glColor3f(0.95, 0.95, 0.95)
            iter_num = (1/self.timeset) / self.time_step
            iter_num = int(iter_num)
            particles = [self.particle]#, self.particle2]
            times = [self.particle.get_time()]#, self.particle2.get_time()]
            ys = [self.particle.get_state()]#, self.particle2.get_state()]
            self.messenger.test_particles(self.time_step, iter_num)
            self.particle = self.messenger.get_test_particle(0)
            # self.particle2 = self.messenger.get_test_particle(1)
            state = self.particle.get_state()
            # state2 = self.particle2.get_state()
            pos = state[:3]
            glColor3f(1., 0., 0.)
            mesh.draw_mesh("SPHERE", start=pos, glDict=self.glDict, size=0.5)
            # mesh.draw_mesh("SPHERE", start=state2[:3], glDict=self.glDict, size=0.5)
            spring_list = self.messenger.get_springs()
            for spring in spring_list:
                connection = spring.get_connection()
                particle1 = connection[0]
                particle2 = connection[1]
                position1 = particle1.get_state()[:3]
                position2 = particle2.get_state()[:3]
                glColor3f(0., 1., 0.)
                mesh.draw_mesh("LINE", start=position1, end=position2)
            # mesh.drawBox()

    def get_character_info(self, character_id, now_frame=None, category="character"):
        character_state_dict = self.messenger.get_character_states(character_id)
        if now_frame is None:
            now_frame = character_state_dict["now_frame"]
        rotmats, positions, velocities = self.messenger.get_character_motion(character_id, now_frame, category=category)

        return character_state_dict, rotmats, positions, velocities

    def draw_IK_target_point(self, history_dict):
        IK_joint_list = history_dict.keys()
        for joint_id in IK_joint_list:
            target_position = history_dict[joint_id]
            mesh.draw_mesh("SPHERE", start=target_position, glDict=self.glDict, size=0.1)

    def draw_limb_IK(self, character_state_dict, rotmats, positions):
        self.draw_character(character_state_dict, rotmats, positions, model_color=(0.9, 0.9, 0.3))

    def draw_jacob_IK(self, character_state_dict, rotmats, positions):
        self.draw_character(character_state_dict, rotmats, positions, model_color=(0.3, 0.3, 0.9))

    def draw_character(self, character_state_dict, rotmats, positions, shadow=False,
                       model_color=(0.8, 0.8, 0.8)):
        is_init_loaded = character_state_dict["init_loaded"]
        is_focused = character_state_dict["focused"]
        is_pinned = character_state_dict["pinned"]

        scale = character_state_dict["scale"]
        parentree = character_state_dict["parentree"]
        childree = character_state_dict["childree"]
        offset_list = character_state_dict["offset_list"]

        # model scale 보정
        glScalef(scale, scale, scale)

        visit_history = set()
        visit_index = 0
        child_index = 0
        is_complete = False
        while not is_complete:
            if visit_index in childree:
                for child in childree[visit_index]:
                    if child not in visit_history:
                        child_index = child
                        break
                    child_index = -1
                if child_index > 0:
                    M = np.identity(4)
                    if is_init_loaded is False:
                        rot_matrix = rotmats[visit_index]
                    else:
                        rot_matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

                    glPushMatrix()
                    glTranslatef(offset_list[visit_index][0], offset_list[visit_index][1],
                                 offset_list[visit_index][2])

                    if visit_index == 0:
                        trans_mat = positions[visit_index]
                        glTranslatef(trans_mat[0], trans_mat[1], trans_mat[2])

                    M[:3, :3] = rot_matrix
                    glMultMatrixf(M.T)

                    glPushMatrix()

                    start = np.array([0, 0, 0])
                    end = np.array(offset_list[child_index])
                    # draw skeleton
                    if self.skeleton_view == True:
                        if shadow == False:
                            if is_focused == True:
                                glColor3f(0.9, 0.3, 0.3)
                            elif is_pinned == True:
                                glColor3f(0.3, 0.9, 0.3)
                            else:
                                glColor3f(model_color[0], model_color[1], model_color[2])
                        mesh.draw_mesh("LINE", start=start, end=end)

                    else:
                        # 재생 조작 선택 대상 윤곽선 그리기
                        if is_focused == True and shadow == False:
                            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                            glColor3f(0.9, 0.3, 0.3)
                        elif is_pinned == True and shadow == False:
                            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                            glColor3f(0.3, 0.9, 0.3)
                        mesh.draw_mesh("BOX", start=start, end=end, glDict=self.glDict, size=scale)
                        # 몸체 그리기
                        if self.frame_view == False and shadow == False:
                            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                            glColor3f(model_color[0], model_color[1], model_color[2])
                            mesh.draw_mesh("BOX", start=start, end=end, glDict=self.glDict, size=scale)
                    glPopMatrix()

                    visit_history.update([visit_index])
                    visit_index = child_index

                # child joint를 모두 방문한 경우
                else:
                    visit_index = parentree[visit_index]
                    if visit_index == -1:
                        is_complete = True
                    else:
                        glPopMatrix()

            # end_point인 경우
            else:
                visit_history.update([visit_index])
                glPopMatrix()
                visit_index = parentree[visit_index]

    def draw_joint(self, positions, type="global"):
        glColor3f(0.5, 0.7, 0.9)

        for joint_index in range(len(positions)):
            joint_pos = positions[joint_index]
            if type == "global":
                if joint_index == 0:
                    glColor3f(0.9, 0., 0.)
                else:
                    glColor3f(0.5, 0.7, 0.9)

                mesh.draw_mesh("SPHERE", start=joint_pos, glDict=self.glDict, size=0.2)
            '''
            # root joint coordinate 기준 좌표값 --> (T_root @ R_root @ 좌표 값 = glboal 좌표값) 식으로 확인
            elif type == "root":
                # R_root @ 좌표 값
                root_coordinate_pos = motion.root_coor_positions[motion.frame][joint_index * 3: joint_index * 3 + 3]
                change2global_rot = motion.rotation_mats[motion.frame][0].reshape(-1, 3) \
                                    @ np.array(root_coordinate_pos).T
                translation = np.identity(4)
                translation[:3, 3] = motion.positions[motion.frame][0:3]
                change2global_rot = np.append(change2global_rot.T, 1)
                # T_root @ (R_root @ 좌표 값)
                change2global = translation @ change2global_rot.T
                glColor3f(0., 0.9, 0.)
                mesh.draw_mesh("SPHERE", start=change2global.T, glDict=self.glDict, size=0.3)
            '''

    def draw_velocity(self, positions, velocities):
        glColor3f(0.5, 0.5, 1.)

        for joint_index in range(len(velocities)):
            now_position = positions[joint_index]
            now_velocity = velocities[joint_index]
            end_position = now_position + (now_velocity * 0.1)

            mesh.draw_mesh("LINE", start=now_position, size=2.0, end=end_position, glDict=self.glDict)

    def OnTime(self, event):
        self.messenger.play_frame()
        self.Refresh()
