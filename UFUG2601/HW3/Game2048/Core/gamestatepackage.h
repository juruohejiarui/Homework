#ifndef GAMESTATEPACKAGE_H
#define GAMESTATEPACKAGE_H

#include "gamestate.h"
#include <string>
#include <deque>

class GameStatePackage {
private:
    std::deque<GameState> states;
    std::string filePath;

    time_t startTime;
public:
    static const int maxStateQueueSize = 200;
    GameStatePackage();
    GameStatePackage(const std::string &_path);

    void load(const std::string &_path);
    void save();

    GameState &getCurrentState();
    void undo();
    void init();
    // operate once and return whether this game is end
    bool Operate(GameOperation _o);

    int getRow();
    int getColumn();
    // this operation may clean the states
    void setRow(int _row);
    // this operation may clean the states
    void setColumn(int _column);

    time_t getStartTime();
};


#endif // GAMESTATEPACKAGE_H
