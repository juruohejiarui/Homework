#include "configuration.h"
#include <fstream>

GameResult::GameResult(int _score, time_t _time, std::pair<int, int> _size, const std::string &_player) {
    this->score = _score, this->time = _time;
    this->player = _player;
    this->size = _size;
}
Configuration::Configuration()
{
    load("default.config");
}

void Configuration::initState() { statePackage.init(); }

std::string readString(std::ifstream &ifs) {
    std::string _res = ""; char _ch;
    while (true) {
        ifs.read(&_ch, sizeof(char));
        if (_ch == '\0') break;
        _res.push_back(_ch);
    }
    return _res;
}

void writeString(std::ofstream &ofs, std::string &_s) {
    ofs.write(_s.c_str(), sizeof(char) * (_s.size() + 1));
}

void Configuration::load(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    filePath = path;
    // create this file if it does not exist
    if (!ifs.good()) {
        player = "Default Player";
        themePath = "Light.theme";
        statePackage = GameStatePackage(path + ".state");
        std::ofstream ofs(path, std::ios::binary);
        ofs.close();
        save();
        ifs.open(path, std::ios::binary);
    }

    player = readString(ifs);
    themePath = readString(ifs);
    statePackage.load(path + ".state");

    int _rk_len; ifs.read((char *)&_rk_len, sizeof(int));
    rankList.clear();
    for (int i = 0; i < _rk_len; i++) {
        int _score; time_t _time;
        std::pair<int, int> _size;
        std::string _player;
        ifs.read((char *)&_score, sizeof(int)), ifs.read((char *)&_time, sizeof(time_t));
        ifs.read((char *)&_size.first, sizeof(int)), ifs.read((char *)&_size.second, sizeof(int));
        _player = readString(ifs);
        rankList.push_back(GameResult(_score, _time, _size, _player));
    }
}

void Configuration::save() {
    std::ofstream ofs(filePath, std::ios::binary);
    writeString(ofs, player);
    writeString(ofs, themePath);
    statePackage.save();
    int _rk_len = rankList.size();
    ofs.write((char *)&_rk_len, sizeof(int));
    for (auto &_result : rankList) {
        ofs.write((char *)&_result.score, sizeof(int)), ofs.write((char *)&_result.time, sizeof(time_t));
        ofs.write((char *)&_result.size.first, sizeof(int)), ofs.write((char *)&_result.size.second, sizeof(int));
        writeString(ofs, _result.player);
    }
    ofs.close();
}

void Configuration::setRow(int _row) {
    statePackage.setRow(_row);
}

void Configuration::setColumn(int _column) {
    statePackage.setColumn(_column);
}

int Configuration::getRow() { return statePackage.getRow(); }
int Configuration::getColumn() { return statePackage.getColumn(); }

void Configuration::setThemePath(const std::string &_path) { themePath = _path; }
const std::string &Configuration::getThemePath() { return themePath; }

const std::string &Configuration::getPlayer() { return player; }
void Configuration::setPlayer(const std::string &_player) { player = _player; }

const std::vector<GameResult> &Configuration::getRankList() { return rankList; }
void Configuration::updateRankList(const GameResult &_result) {
    auto iter = rankList.begin();
    while (iter != rankList.end() && iter->score > _result.score) iter++;
    rankList.insert(iter, _result);
    if (rankList.size() > 20) rankList.pop_back();
}

GameStatePackage &Configuration::getStatePackage() { return statePackage; }

