#include "gameboard-cli.h"
#include "io-cli.h"
#include "configuration.h"

enum ViewState {
    Playing, RankList, Welcome, Pause, SetPlayer,
} currentView;

enum ValidKey {
    A, D, Q, S, U, W, Z, Left, Right, Up, Down, Waiting
};
bool keepGoing;
static Configuration configuration;

static char textBuffer[1024];

void updateView_Playing() {
    cprint("Current Player : ");
    sprintf(textBuffer, "%s\n", configuration.getPlayer().c_str());
    cprint(textBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("Score : ");
    sprintf(textBuffer, "%d\n", configuration.getStatePackage().getCurrentState().getScore());
    cprint(textBuffer, CLI_COLOR_BLUE | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);

    int _row = configuration.getRow(), _col = configuration.getColumn();
    cprint("   ");
    for (int j = 1; j <= _col; j++) {
        sprintf(textBuffer, "%7d", j);
        cprint(textBuffer, CLI_COLOR_RED);
    }
    cprint("\n");
    for (int i = 0; i < _row; i++) {
        sprintf(textBuffer, "%3d", i + 1);
        cprint(textBuffer, CLI_COLOR_RED);
        for (int j = 0; j < _col; j++) {
            int _dn = (1 << configuration.getStatePackage().getCurrentState()[i][j]);
            if (_dn == 1) sprintf(textBuffer, "       %c", (j == _col - 1 ? '\n' : '\0'));
            else sprintf(textBuffer, "%7d%c", _dn, (j == _col - 1 ? '\n' : '\0'));
            int _c = CLI_COLOR_WHITE;
            if (_dn >= 4096) _c = CLI_COLOR_RED | CLI_COLOR_RED;
            else if (_dn >= 1024) _c = CLI_COLOR_GREEN | CLI_COLOR_INTENSITY;
            else if (_dn >= 128) _c = CLI_COLOR_GREEN;
            cprint(textBuffer, _c);
        }
    }
    cprint("\n\n");

    cprint("[");
    cprint("ARROW / {ASDW}", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Start [");
    cprint("Z", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Undo [");
    cprint("U", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Change User Name [");
    cprint("R", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Rank List [");
    cprint("Q", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Pause\n");
}
void updateView_RankList() {

}

void updateView_Welcome() {
    cprint("Welcome!!!\n", CLI_COLOR_RED | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK);
    
    cprint("[");
    cprint("A", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Start [");
    cprint("U", CLI_COLOR_GREEN | CLI_COLOR_INTENSITY, CLI_COLOR_BLACK, true);
    cprint("] Change User Name\n");
}
void updateView_Pause() {

}
void updateView() {
    system("clear");
    switch (currentView) {
        case ViewState::Pause: updateView_Pause(); break;
        case ViewState::Playing: updateView_Playing(); break;
        case ViewState::RankList: updateView_RankList(); break;
        case ViewState::Welcome: updateView_Welcome(); break;
    }
}

static int keyBuffer[4], keyBufferSize;

ValidKey getKey() {
    int _key = keyboardRead();
    printf("%d buf = {", _key);
    for (int i = 0; i < keyBufferSize; i++) printf("%d ", keyBuffer[i]);
    printf("}\n");
    switch (_key) {
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
        case 115:
             if (!keyBufferSize) return ValidKey::S;
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

void inputHandler_Pause(ValidKey _key) {
    
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
    }
}

void inputHandler_RankList(ValidKey _key) {
    
}

void inputHandler_Welcome(ValidKey _key) {
    
}

void inputHandler_SetPlayer(int _key) {

}

void inputHandler() {
    // 这个界面比较特殊，要获取所有类型的输入
    if (currentView == ViewState::SetPlayer) inputHandler_SetPlayer(keyboardRead());
    ValidKey _key = getKey();
    if (_key == ValidKey::Waiting) return ;
    switch(currentView) {
        case ViewState::Pause: inputHandler_Pause(_key); break;
        case ViewState::Playing: inputHandler_Playing(_key); break;
        case ViewState::RankList: inputHandler_RankList(_key); break;
        case ViewState::Welcome: inputHandler_Welcome(_key); break;
    }
}
void exec2048() {
    configuration.load("default.config");
    keepGoing = true;
    currentView = ViewState::Playing;
    while (keepGoing) {
        updateView();
        inputHandler();
    }
    configuration.save();
}