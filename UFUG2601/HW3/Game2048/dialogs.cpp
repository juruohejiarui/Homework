#include "dialogs.h"

void showWarning(const std::string &_title, const std::string &_message) {
    QMessageBox _msgbox;
    _msgbox.setText(_message.c_str());
    _msgbox.setWindowTitle(_title.c_str());
    _msgbox.show();
}

bool ensureAbort() {
    QMessageBox msgbx;
    msgbx.setText("This operation will abort this game, continue?");
    msgbx.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
    msgbx.setDefaultButton(QMessageBox::Ok);
    int res = msgbx.exec();
    if (res == QMessageBox::Ok) return true;
    else return false;
}