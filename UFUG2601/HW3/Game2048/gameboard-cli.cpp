#include "gameboard-cli.h"
#include "io-cli.h"
#include "configuration.h"

enum ViewState {
    Playing, RankList, Welcome, Pause, InputDialogBox, DialogBox, Unknown
} currentView, lastView;
enum DialogBoxType {
    Ok, OkAndCancel, Input,
};

enum ValidKey {
    A, D, Q, R, S, U, W, Z, Left, Right, Up, Down, Enter, Waiting
};
bool keepGoing;
static Configuration configuration;

static char stringBuffer[1024], dialogText[1024];
DialogBoxType dialogBoxType;
int dialogResult, dialogFlag[15];

/*
ViewState::Playing:
    1. change player name
    2. warning after change player name
ViewState::Pause:
    1. abort current game before start a new game
    2. abort current game before change board size
    3. change board size
    4. change player name
    5. warning after change player name
    6. warning after change board size
*/

#pragma region Update View
void printOperationHint(const char *_k, const char *_d) {
    cprint("[");
    cprint(_k, CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    sprintf(stringBuffer, "] %s ", _d);
    cprint(stringBuffer);
} 

void updateView_Playing() {
    cprint("[BOARD]\n", CLI_COLOR_BLUE, CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_WHITE);
    cprint("Current Player : ");
    sprintf(stringBuffer, "%s\n", configuration.getPlayer().c_str());
    cprint(stringBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("Score : ");
    sprintf(stringBuffer, "%d\n", configuration.getStatePackage().getCurrentState().getScore());
    cprint(stringBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);

    int _row = configuration.getRow(), _col = configuration.getColumn();
    cprint("   ");
    for (int j = 1; j <= _col; j++) {
        sprintf(stringBuffer, "%7d", j);
        cprint(stringBuffer, CLI_COLOR_RED);
    }
    cprint("\n");
    for (int i = 0; i < _row; i++) {
        sprintf(stringBuffer, "%3d", i + 1);
        cprint(stringBuffer, CLI_COLOR_RED);
        for (int j = 0; j < _col; j++) {
            int _dn = (1 << configuration.getStatePackage().getCurrentState()[i][j]);
            if (_dn == 1) sprintf(stringBuffer, "       %c", (j == _col - 1 ? '\n' : '\0'));
            else sprintf(stringBuffer, "%7d%c", _dn, (j == _col - 1 ? '\n' : '\0'));
            int _c = CLI_COLOR_WHITE;
            if (_dn >= 4096) _c = CLI_COLOR_RED | CLI_COLOR_RED;
            else if (_dn >= 1024) _c = CLI_COLOR_GREEN | CLI_COLOR_INTENSITY;
            else if (_dn >= 128) _c = CLI_COLOR_GREEN;
            cprint(stringBuffer, _c);
        }
    }
    cprint("\n\n");

    printOperationHint("ARROW / {ASDW}", "Start");
    printOperationHint("Z", "Undo");
    printOperationHint("Q", "Pause");
    putchar('\n');
}

static int rankListScroll;

void updateView_RankList() {

}

void updateView_Welcome() {
    cprint("Welcome!!!\n", CLI_COLOR_RED | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK);
    
    printOperationHint("ARROW / {ASDW} / Enter", "Start / Continue");
    printOperationHint("R", "Rank List");
}
void updateView_Pause() {
    cprint("[PAUSE]\n", CLI_COLOR_BLUE, CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_WHITE);
    cprint("Current Player : ");
    sprintf(stringBuffer, "%s\n", configuration.getPlayer().c_str());
    cprint(stringBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("Score : ");
    sprintf(stringBuffer, "%d\n\n", configuration.getStatePackage().getCurrentState().getScore());
    cprint(stringBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    
    printOperationHint("Enter", "Continue");
    putchar('\n');
    printOperationHint("  Z  ", "Start a New Game");
    putchar('\n');
    printOperationHint("  U  ", "Change Player Name");
    putchar('\n');
    printOperationHint("ARROW", "Change Board Size");
    putchar('\n');
    printOperationHint("  R  ", "Rank List");
    putchar('\n');
    printOperationHint("  Q  ", "Exit");
    putchar('\n');
}


void update_dialogBox() {
    cprint(dialogText);
    putchar('\n');
    switch (dialogBoxType) {
        case DialogBoxType::Ok:
            printOperationHint("Q / Enter", "OK");
            break;
        case DialogBoxType::OkAndCancel:
            printOperationHint("Z", "OK");
            printOperationHint("Q / Enter", "Cancel");
            break;
    }
    putchar('\n');
}

static char inputDialogContent[1024];
static int inputDialogContentLength;
void updateView_InputDialogBox() {
    cprint(dialogText);
    putchar('\n');
    memcpy(stringBuffer, inputDialogContent, inputDialogContentLength * sizeof(char));
    stringBuffer[inputDialogContentLength] = '\0';
    cprint(stringBuffer, CLI_COLOR_RED | CLI_COLOR_GREEN | CLI_COLOR_INTENSITY);
    putchar('\n'), putchar('\n');
    printOperationHint("Enter", "OK");
    printOperationHint("ESC", "Cancel");
    putchar('\n');
}

void updateView() {
    system("clear");
    switch (currentView) {
        case ViewState::Pause: updateView_Pause(); break;
        case ViewState::Playing: updateView_Playing(); break;
        case ViewState::RankList: updateView_RankList(); break;
        case ViewState::Welcome: updateView_Welcome(); break;
        case ViewState::DialogBox: update_dialogBox(); break;
        case ViewState::InputDialogBox: updateView_InputDialogBox(); break;
    }
}

void switchView(ViewState _view) {
    if (_view == ViewState::DialogBox || _view == ViewState::InputDialogBox)
        dialogResult = -1;
    currentView = _view;
    rankListScroll = 0;
}

void showDialog(ViewState _dialog) {
    lastView = currentView;
    currentView = _dialog;
    dialogResult = -1;
}

void closeDialog() {
    currentView = lastView;
    lastView = ViewState::Unknown;
    switchView(currentView);
}

#pragma endregion

#pragma region Keyboard Buffer
static int keyBuffer[4], keyBufferSize;

ValidKey getKey() {
    int _key = keyboardRead();
    printf("%d buf = {", _key);
    for (int i = 0; i < keyBufferSize; i++) printf("%d ", keyBuffer[i]);
    printf("}\n");
    switch (_key) {
        case 13:
            keyBufferSize = 0;
            return ValidKey::Enter;
        case 27:
            keyBufferSize = 0;
            keyBuffer[keyBufferSize++] = _key;
            return ValidKey::Waiting;
        case 91:
            if (keyBufferSize == 1 && keyBuffer[0] == 27) {
                keyBuffer[keyBufferSize++] = 91;
                return ValidKey::Waiting;
            } else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 65:
            if (keyBufferSize == 2) {
                keyBufferSize = 0;
                return ValidKey::Up;
            } else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 66:
            if (keyBufferSize == 2) {
                keyBufferSize = 0;
                return ValidKey::Down;
            } else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 67:
            if (keyBufferSize == 2) {
                keyBufferSize = 0;
                return ValidKey::Right;
            } else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 68:
            if (keyBufferSize == 2) {
                keyBufferSize = 0;
                return ValidKey::Left;
            } else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 97:
            if (!keyBufferSize) return ValidKey::A;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 100:
             if (!keyBufferSize) return ValidKey::D;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 113:
            if (!keyBufferSize) return ValidKey::Q;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 114:
            if (!keyBufferSize) return ValidKey::R;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 115:
            if (!keyBufferSize) return ValidKey::S;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 117:
            if (!keyBufferSize) return ValidKey::U;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 119:
            if (!keyBufferSize) return ValidKey::W;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
        case 122:
            if (!keyBufferSize) return ValidKey::Z;
            else {
                keyBufferSize = 0;
                return ValidKey::Waiting;
            }
    }
    keyBufferSize = 0;
    return ValidKey::Waiting;
}
#pragma endregion

#pragma region Input Handler
void inputHandler_Pause(ValidKey _key) {
    switch (_key) {
        case ValidKey::Enter:
            switchView(ViewState::Playing);
            break;
        case ValidKey::Z:
            if (!configuration.getStatePackage().getCurrentState().end()) {
                dialogFlag[ViewState::Pause] = 1;
                sprintf(dialogText, "This operation will abort the current game, continue?");
                dialogBoxType = DialogBoxType::OkAndCancel;
                showDialog(ViewState::DialogBox);
            } else {
                configuration.updateRankList();
                configuration.initState();
                switchView(ViewState::Playing);
            }
            break;
        case ValidKey::U:
            dialogFlag[ViewState::Pause] = 4;
            sprintf(dialogText, "Input the new player name (<= 20)");
            showDialog(ViewState::InputDialogBox);
            break;
        case ValidKey::R:
            switchView(ViewState::RankList);
            break;
        case ValidKey::Left:
        case ValidKey::Right:
        case ValidKey::Up:
        case ValidKey::Down:
            if (!configuration.getStatePackage().getCurrentState().end()) {
                dialogFlag[ViewState::Pause] = 2;
                sprintf(dialogText, "This operation will abort the current game, continue?");
                dialogBoxType = DialogBoxType::OkAndCancel;
                showDialog(ViewState::DialogBox);
            } else {
                configuration.updateRankList();
                dialogFlag[ViewState::Pause] = 3;
                sprintf(dialogText, "Input the new board size: <Row> <Column>");
                showDialog(ViewState::DialogBox);
            }
            break;
    }
}

void inputHandler_Playing(ValidKey _key) {
    bool _end = false;
    switch (_key) {
        case ValidKey::Left:
        case ValidKey::A:
            if (configuration.getStatePackage().getCurrentState().checkValid(GameOperation::Left))
                _end = configuration.getStatePackage().Operate(GameOperation::Left);
            break;
        case ValidKey::Right:
        case ValidKey::D:
            if (configuration.getStatePackage().getCurrentState().checkValid(GameOperation::Right))
                _end = configuration.getStatePackage().Operate(GameOperation::Right);
            break;
        case ValidKey::Up:
        case ValidKey::W:
            if (configuration.getStatePackage().getCurrentState().checkValid(GameOperation::Up))
                _end = configuration.getStatePackage().Operate(GameOperation::Up);
            break;
        case ValidKey::Down:
        case ValidKey::S:
            if (configuration.getStatePackage().getCurrentState().checkValid(GameOperation::Down))
                _end = configuration.getStatePackage().Operate(GameOperation::Down);
            break;
        case ValidKey::Z:
            configuration.getStatePackage().undo();
            break;
        case ValidKey::Q:
            configuration.updateRankList();
            switchView(ViewState::Pause);
            break;
    }
    if (_end) switchView(ViewState::Pause);
}

void inputHandler_RankList(ValidKey _key) {
    switch (_key) {

    }
}

void inputHandler_Welcome(ValidKey _key) {
    
}

void inputHandler_InputDialog(int _key) {
}

void inputHandler_DialogBox(ValidKey _key) {
    if (dialogBoxType == DialogBoxType::Ok) {
        if (_key == ValidKey::Enter || _key == ValidKey::Q)
            dialogResult = 0, closeDialog();
    } else {
        if (_key == ValidKey::Enter || _key == ValidKey::Q)
            dialogResult = 0, closeDialog();
        else if (_key == ValidKey::Z)
            dialogResult = 1, closeDialog();
    }
}

void dialogResultHandler_Pause() {
    int &_flag = dialogFlag[ViewState::Pause];
    if (_flag == 1) {
        if (dialogResult == 1) configuration.initState();
        switchView(ViewState::Playing);
    } else if (_flag == 2) {
        if (dialogResult == 1) {
            configuration.updateRankList();
            _flag = 3;
            sprintf(dialogText, "Input the new board size: <Row> <Column>");
            showDialog(ViewState::DialogBox);
        } else switchView(ViewState::Playing);
    } else if (_flag == 3) {
        if (dialogResult == 1) {
            std::string _new_size_str(inputDialogContent);
            int _nrow = 0, _ncol = 0;
            if (_new_size_str.size() != 3) goto INVALID_SIZE;
            if ('0' > std::min(_new_size_str[0], _new_size_str[2]) || '9' < std::max(_new_size_str[0], _new_size_str[2])
                || _new_size_str[1] != ' ') goto INVALID_SIZE;
            VALID_SIZE:
                _nrow = _new_size_str[0] - '0', _ncol = _new_size_str[2] - '0';
                configuration.updateRankList();
                configuration.setRow(_nrow);
                configuration.setColumn(_ncol);
                switchView(ViewState::Playing);
                goto END;
            INVALID_SIZE:
                _flag = 6;
                sprintf(dialogText, "Valid Board Size\n, the number of row and column must be in [1, 8]");
                showDialog(ViewState::DialogBox);
            END:
        }
    } else if (_flag == 4) {
        if (dialogResult == 1) {
            std::string _new_name(inputDialogContent);
            if (_new_name.size() == 0 || _new_name.size() > 20) {
                _flag = 5;
                memcpy(dialogText, "Invalid Player Name\nThe length of player name must be in [1, 20]", 65);
                showDialog(ViewState::DialogBox);
            } else configuration.setPlayer(std::string(inputDialogContent));
        }
    } else if (_flag == 5 || _flag == 6) {
        switchView(ViewState::Playing);
    }
    _flag = 0;
}

void dialogResultHandler_Playing() {
    int &_flag = dialogFlag[ViewState::Playing];
    _flag = 0;
}

void inputHandler() {
    // use the result of the messagebox
    if (dialogResult != -1) {
        switch (currentView) {
            case ViewState::Pause: dialogResultHandler_Pause(); break;
            case ViewState::Playing: dialogResultHandler_Playing(); break;
        }
        dialogResult = -1;
        return ;
    }
    // 这个界面比较特殊，要获取所有类型的输入
    if (currentView == ViewState::InputDialogBox) {
        inputHandler_InputDialog(keyboardRead());
        return ;
    }
    ValidKey _key = getKey();
    if (_key == ValidKey::Waiting) return ;
    switch(currentView) {
        case ViewState::Pause: inputHandler_Pause(_key); break;
        case ViewState::Playing: inputHandler_Playing(_key); break;
        case ViewState::RankList: inputHandler_RankList(_key); break;
        case ViewState::Welcome: inputHandler_Welcome(_key); break;
        case ViewState::DialogBox: inputHandler_DialogBox(_key); break;
    }
}
#pragma endregion

void exec2048() {
    configuration.load("default.config");
    keepGoing = true;
    currentView = ViewState::Playing;
    dialogResult = -1;
    while (keepGoing) {
        updateView();
        inputHandler();
    }
    configuration.save();
}