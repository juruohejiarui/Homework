#ifndef __IO_CLI_H_
#define __IO_CLI_H_

void keyboardInit();
int keyboardRead();
void keyboardClose();

#ifdef __linux__
#define CLI_COLOR_BLACK 0x00
#define CLI_COLOR_RED 0x01
#define CLI_COLOR_GREEN 0x02
#define CLI_COLOR_BLUE 0x04
#define CLI_COLOR_WHITE 0x07

#define CLI_COLOR_INTENSITY 0x08
#elif __APPLE__
#define CLI_COLOR_BLACK 0x00
#define CLI_COLOR_RED 0x01
#define CLI_COLOR_GREEN 0x02
#define CLI_COLOR_BLUE 0x04
#define CLI_COLOR_WHITE 0x07

#define CLI_COLOR_INTENSITY 0x08
#elif _WIN32
#define CLI_COLOR_BLACK 0x00
#define CLI_COLOR_RED 0x04
#define CLI_COLOR_GREEN 0x02
#define CLI_COLOR_BLUE 0x01
#define CLI_COLOR_WHITE 0x07

#define CLI_COLOR_INTENSITY 0x08
#endif

void cprint(const char* info, short _fcol = CLI_COLOR_WHITE, short _bcol = 0, bool _ul = false);

void cleanScreen();
#endif