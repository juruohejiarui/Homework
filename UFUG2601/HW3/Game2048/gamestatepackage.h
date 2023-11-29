#ifndef GAMESTATEPACKAGE_H
#define GAMESTATEPACKAGE_H

#include "gamestate.h"
#include <string>

class GameStatePackage {
private:
    std::vector<GameState> states;
    std::string filePath;
public:
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
};


#endif // GAMESTATEPACKAGE_H
