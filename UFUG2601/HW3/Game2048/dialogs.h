#ifndef DIALOGS_H
#define DIALOGS_H

#include <QMessageBox>
#include <QInputDialog>

void showWarning(const std::string &_title, const std::string &_message);
bool ensureAbort();

#endif // DIALOGS_H
