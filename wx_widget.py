#기능 구현 관련
#1. combo box widget으로 read된 motion file을 선택 (focused)
#2. focuse된 파일은 기본적으로 재생 관련 기능 제어 대상이 되며 추가적으로 삭제 및 재생 구간 설정, 슬라이더 조작 대상이 됨
#3. focused 이외에도 다중 파일의 재생 관련 기능을 일부 허용 (pinned)
#4. pinned 된 파일은 재생 관련 기능의 제어 대상 (재생 관련 조작의 대상이 됨)
#5. 재생 관련 조작은 space(재생, 일시정지), 방향키 좌/우(되감기, 앞으로 감기), 방향키 상/하(재생 배속 조절 - 임시), 미정(정지) 가 해당됨
#6. 구간 반복 설정 후 refresh 버튼을 눌러 구간 반복 지정하도록 함
#7. 구간 설정은 낮은 순번의 frame 부터 시작할 수도 있고 큰 순번의 frame 부터 시작해서 낮은 순번 frame에서 끝나도록 할 수도 있음
#(예 - 10번째 frame에서 시작 -> 100번째 frame에서 종료 /
# 100번째 frame에서 시작 -> 120번째 frame이 마지막 -> 0으로 돌아감 -> 10번째 프레임에서 종료)
#8. slider가 구간 반복 설정 시 현재 위치가 구간 밖이면 구간 시작 부분으로 가도록 설정, 구간 설정된 부분 밖으로 슬라이더 이동 불가하도록 설정
#9. combo box 옆 x 버튼은 해당 model 삭제 버튼

import wx


class FileDropTarget(wx.FileDropTarget):
    def __init__(self, frame, messenger):
        super().__init__()
        self.frame = frame
        self.messenger = messenger

    def OnDropFiles(self, x, y, filename):
        self.messenger.read_file(filename)
        return True


class Play_Slider(wx.Slider):
    def __init__(self, panel, messenger):
        super().__init__(panel, value=0, maxValue=1, style=wx.SL_MIN_MAX_LABELS)
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_SCROLL_CHANGED, self.scroll)
        self.Bind(wx.EVT_SCROLL_THUMBTRACK, self.scroll)

    def scroll(self, event):
        slider = event.GetEventObject()
        value = slider.GetValue()
        value = self.messenger.change_frame(value)


class Start_Frame(wx.TextCtrl):
    def __init__(self, panel, messenger):
        super().__init__(panel, size = (100, 30), style = wx.TE_PROCESS_ENTER|wx.TE_RIGHT)
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_TEXT_ENTER, self.change_value)

    def change_value(self, event):
        value = int(self.GetValue())
        value = self.messenger.change_frame_state("start_frame", value)
        self.SetValue(str(value))


class End_Frame(wx.TextCtrl):
    def __init__(self, panel, messenger):
        super().__init__(panel, size = (100, 30), style = wx.TE_PROCESS_ENTER|wx.TE_RIGHT)
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_TEXT_ENTER, self.change_value)

    def change_value(self, event):
        value = int(self.GetValue())
        value = self.messenger.change_frame_state("end_frame", value)
        self.SetValue(str(value))


class Frame_typer(wx.TextCtrl):
    def __init__(self, panel, messenger):
        super().__init__(panel, size = (100,30), style = wx.TE_PROCESS_ENTER|wx.TE_RIGHT)
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_TEXT_ENTER, self.change_frame)

    def change_frame(self, event):
        value = int(self.GetValue())
        value = self.messenger.change_frame(value)


# focused motion model 반복 구간 부분 data bvh(예정) 형식으 export
class Export_Button(wx.Button):
    def __init__(self, panel, messenger):
        super().__init__(panel, label='Export', size=(70, 30))
        self.frame = panel.GetParent()
        self.messenger = messenger


class Joint_Button(wx.Button):
    def __init__(self, panel, messenger):
        super().__init__(panel, label='Show Joint', size=(100, 30))
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_BUTTON, self.click)

    def click(self, event):
        state = self.messenger.change_joint_view_state()
        self.change_label(state)

    def change_label(self, state):
        if state:
            self.SetLabel("Hide Joint")
        else:
            self.SetLabel("Show Joint")


class Velocity_Button(wx.Button):
    def __init__(self, panel, messenger):
        super().__init__(panel, label='Show Velocity', size=(100, 30))
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_BUTTON, self.click)

    def click(self, event):
        state = self.messenger.change_velocity_view_state()
        self.change_label(state)

    def change_label(self, state):
        if state:
            self.SetLabel("Hide Joint")
        else:
            self.SetLabel("Show Joint")


class Start_Button(wx.Button):
    def __init__(self, panel, messenger):
        super().__init__(panel, label="▶", size= (30, 30))
        self.frame = panel.GetParent()
        self.messenger = messenger
        self.Bind(wx.EVT_BUTTON, self.play)

    def play(self, event):
        state = self.messenger.change_play_state()
        self.change_label(state)

    def change_label(self, state):
        if state:
            self.SetLabel("∥")
        else:
            self.SetLabel("▶")
