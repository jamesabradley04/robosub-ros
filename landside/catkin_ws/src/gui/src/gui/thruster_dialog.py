
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QDialog
from python_qt_binding.QtCore import QThread

import rospy
import resource_retriever as rr

from custom_msgs.msg import ThrusterSpeeds
from gui.publisher_object import PublisherObject

class ThrusterDialog(QDialog):

    def __init__(self):
        super(ThrusterDialog, self).__init__()
        
        ui_file = rr.get_filename('package://gui/resource/ThrusterDialog.ui', use_protocol=False)
        loadUi(ui_file, self)

        self.pub = rospy.Publisher('/offboard/thruster_speeds', ThrusterSpeeds, queue_size=3)
        self.msg = ThrusterSpeeds(speeds=[0, 0, 0, 0, 0, 0, 0, 0])
        self.pub_obj = PublisherObject(self.pub, self.msg)

        self.sliders = [
            self.tfr_slider,
            self.tfl_slider,
            self.tbr_slider,
            self.tbl_slider,
            self.bfr_slider,
            self.bfl_slider,
            self.bbr_slider,
            self.bbl_slider
        ]

        self.editors = [
            self.tfr_edit,
            self.tfl_edit,
            self.tbr_edit,
            self.tbl_edit,
            self.bfr_edit,
            self.bfl_edit,
            self.bbr_edit,
            self.bbl_edit
        ]

        for i in range(8):
            self.editors[i].valueChanged.connect(lambda val, index=i: self.editor_changed(index, val))
            self.sliders[i].valueChanged.connect(lambda val, index=i: self.slider_changed(index, val))

    def show(self):
        super(ThrusterDialog, self).show()
        for i in range(8):
            self.editors[i].setValue(0)
            self.sliders[i].setValue(0)
        self.msg = ThrusterSpeeds(speeds=[0, 0, 0, 0, 0, 0, 0, 0])
        self.pub_obj = PublisherObject(self.pub, self.msg)
        self.thread = QThread()
        self.pub_obj.moveToThread(self.thread)
        self.thread.started.connect(self.pub_obj.run)
        self.thread.start()

    def slider_changed(self, index, val):
        rospy.loginfo(f"Slider Index: {index} value {val}")
        self.editors[index].setValue(val)
        self.update_msg(index, val)

    def editor_changed(self, index, val):
        rospy.loginfo(f"Editor Index: {index} value {val}")
        self.sliders[index].setValue(val)
        self.update_msg(index, val)

    def update_msg(self, index, new_val):
        self.msg.speeds[index] = new_val
        self.pub_obj.update_msg(self.msg)

    def reject(self):
        self.pub_obj.stop()
        self.thread.quit()
        self.thread.wait()
        super(ThrusterDialog, self).reject()

