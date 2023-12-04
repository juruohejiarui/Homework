#include "io-cli.h"
#include <cstdio>

#ifdef __linux__
#include <termios.h>
int keyboardRead(void) {
    struct termios tm, tm_old;
    int fd = 0, ch;
 
    if (tcgetattr(fd, &tm) < 0) {//保存现在的终端设置
        return -1;
    }
 
    tm_old = tm;
    cfmakeraw(&tm);//更改终端设置为原始模式，该模式下所有的输入数据以字节为单位被处理
    if (tcsetattr(fd, TCSANOW, &tm) < 0) {//设置上更改之后的设置
        return -1;
    }

     ch = getchar();
    if (tcsetattr(fd, TCSANOW, &tm_old) < 0) {//更改设置为最初的样子
        return -1;
    }

    return ch;
}

void setColor(short _fcol, short _bcol) {
    printf("\033[0m");
    if (_fcol & 0x08) printf("\033[1m");
    printf("\033[3%c;4%cm", (_fcol & 0x7) + '0', (_bcol & 0x7) + '0');
}


void cprint(const char* info, short _fcol, short _bcol, bool _ul) {
    setColor(_fcol, _bcol);
    if (_ul) printf("\033[4m");
    printf("%s", info);
}
#endif

#ifdef __APPLE__
#include <termios.h>
int keyboardRead(void) {
    struct termios tm, tm_old;
    int fd = 0, ch;
 
    if (tcgetattr(fd, &tm) < 0) {//保存现在的终端设置
        return -1;
    }
 
    tm_old = tm;
    cfmakeraw(&tm);//更改终端设置为原始模式，该模式下所有的输入数据以字节为单位被处理
    if (tcsetattr(fd, TCSANOW, &tm) < 0) {//设置上更改之后的设置
        return -1;
    }

     ch = getchar();
    if (tcsetattr(fd, TCSANOW, &tm_old) < 0) {//更改设置为最初的样子
        return -1;
    }

    return ch;
}

void setColor(short _fcol, short _bcol) {
    printf("\033[0m");
    if (_fcol & 0x08) printf("\033[1m");
    printf("\033[3%c;4%cm", (_fcol & 0x7) + '0', (_bcol & 0x7) + '0');
}


void cprint(const char* info, short _fcol, short _bcol, bool _ul) {
    setColor(_fcol, _bcol);
    if (_ul) printf("\033[4m");
    printf("%s", info);
}
#endif 

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
int keyboardRead(void) { return getch(); }

void setColor(short _fcol, short _bcol, bool _ul) {
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);//取标准输入输出句柄 
	SetConsoleTextAttribute(handle, (int)_fcol & (((int)_bcol) << 4) & (((int)ul) << 8));//字符与 color相同
}

void cprint(const char *info, short _fol, short _bcol, bool _ul) {
    setColor(_fcol, _bcol, _ul);
    printf("%s", info);
}
#endif
