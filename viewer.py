import wx
from abc import *

from messenger import *
import wx_widget
import canvas


class Viewer(wx.Frame):
    def __init__(self, parent, title, size=(800, 900)):
        super().__init__(parent, title=title, size=size)
        self.panel = None
        self.canvas = None
        self.file_reader = None
        self.messenger = None

    @abstractmethod
    def init_ui(self):
        pass


class MainWindow(Viewer):
    def __init__(self, parent, title, size=(1400, 940)):
        super().__init__(parent, title=title, size=size)
        self.init_ui()
        self.Show()

    def init_ui(self):
        self.messenger = Messenger()
        self.panel = wx.Panel(self)
        mainbox = wx.BoxSizer(wx.HORIZONTAL)
        playerbox = wx.BoxSizer(wx.VERTICAL)
        menubox = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetBackgroundColour('#dcdcdc')

        mainbox.Add(playerbox)
        mainbox.Add(menubox)

        # canvas space
        horizontalbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.canvas = canvas.Canvas(self.panel, self.messenger)
        self.canvas.SetMinSize((800, 800))
        self.messenger.set_canvas(self.canvas)

        horizontalbox1.Add(self.canvas, wx.ALIGN_TOP | wx.ALIGN_CENTER)
        playerbox.Add(horizontalbox1, 1, wx.EXPAND | wx.ALIGN_TOP | wx.ALL, 10)

        # slider 출력, check box - pin check: pin 설정 해 놓으면 해당 model은 재생 관련 control 같이 수행
        # pin check된 model도 highlighting
        horizontalbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.play_slider = wx_widget.Play_Slider(self.panel, self.messenger)
        self.frame_typer = wx_widget.Frame_typer(self.panel, self.messenger)
        self.start_button = wx_widget.Start_Button(self.panel,self.messenger)

        self.messenger.set_slider(self.play_slider)
        self.messenger.set_frame_typer(self.frame_typer)
        self.messenger.set_start_button(self.start_button)

        horizontalbox3.Add(self.play_slider, wx.EXPAND)
        horizontalbox3.Add(self.frame_typer, flag=wx.LEFT, border=10)
        horizontalbox3.Add(self.start_button, flag=wx.LEFT, border=10)
        playerbox.Add(horizontalbox3, 2, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # 재생 시작 frame, 재생 종료 frame 설정, frame 재생 영역 설정 적용(refresh), 모델 bvh로 내보내기
        horizontalbox4 = wx.BoxSizer(wx.HORIZONTAL)

        self.joint_button = wx_widget.Joint_Button(self.panel, self.messenger)
        self.velocity_button = wx_widget.Velocity_Button(self.panel, self.messenger)
        self.start_label = wx.StaticText(self.panel, label="start frame: ")
        self.start_frame = wx_widget.Start_Frame(self.panel, self.messenger)
        self.end_label = wx.StaticText(self.panel, label="end frame: ")
        self.end_frame = wx_widget.End_Frame(self.panel, self.messenger)
        self.export_button = wx_widget.Export_Button(self.panel, self.messenger)

        self.messenger.set_joint_button(self.joint_button)
        self.messenger.set_velocity_button(self.velocity_button)
        self.messenger.set_end_frame(self.end_frame)
        self.messenger.set_start_frame(self.start_frame)

        horizontalbox4.Add(self.joint_button, flag=wx.ALIGN_LEFT | wx.LEFT)
        horizontalbox4.Add(self.velocity_button, flag=wx.Bottom | wx.LEFT, border=5)
        horizontalbox4.Add(self.start_label, flag=wx.Bottom | wx.LEFT, border=20)
        horizontalbox4.Add(self.start_frame, flag=wx.LEFT | wx.Bottom, border=5)
        horizontalbox4.Add(self.end_label, flag=wx.LEFT | wx.Bottom, border=10)
        horizontalbox4.Add(self.end_frame, flag=wx.LEFT | wx.Bottom, border=5)
        horizontalbox4.Add(self.export_button, flag=wx.LEFT | wx.Bottom, border=5)
        playerbox.Add(horizontalbox4, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)

        playerbox.Add((-1, 10))

        self.panel.SetSizer(mainbox)

        self.file_reader = wx_widget.FileDropTarget(self, self.messenger)
        self.SetDropTarget(self.file_reader)


def main():
    app = wx.App()
    MainWindow(None, title='bvh_player')
    app.MainLoop()


if __name__ == '__main__':
    main()