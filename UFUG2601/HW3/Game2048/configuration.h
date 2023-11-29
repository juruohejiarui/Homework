#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "gamestatepackage.h"
#include <string>
#include <ctime>

struct GameResult {
    std::string player;
    time_t time;
    int score;
    std::pair<int, int> size;

    GameResult(int _score, time_t _time, std::pair<int, int> _size, const std::string &_player);
};

class Configuration {
    std::string player;
    GameStatePackage statePackage;
    std::vector<GameResult> rankList;

    std::string filePath;
    std::string themePath;

public:

    Configuration();

    void initState();

    // load the configuration from file
    void load(const std::string &path);
    void save();

    int getRow();
    int getColumn();
    // this operation may clean the state
    void setRow(int _row);
    // this operation may clean the state
    void setColumn(int _col);

    void setThemePath(const std::string &_path);
    const std::string &getThemePath();


    const std::string &getPlayer();
    void setPlayer(const std::string &_player);

    const std::vector<GameResult> &getRankList();
    void updateRankList(const GameResult &_result);

    GameStatePackage &getStatePackage();
};


#endif // CONFIGURATION_H
