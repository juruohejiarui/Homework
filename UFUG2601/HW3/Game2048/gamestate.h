#ifndef GAMESTATE_H
#define GAMESTATE_H

#include <vector>
#include <stack>

enum class GameOperation {
    Left, Right, Up, Down
};

class GameState {
private:
    int row, column, score;
    std::vector< std::vector<int> > data;

    bool checkValidLeft();
    bool checkValidRight();
    bool checkValidUp();
    bool checkValidDown();

    // the operations
    void updateStateLeft();
    void updateStateRight();
    void updateStateUp();
    void updateStateDown();

public:
    int getRow();
    int getColumn();
    std::vector<int> &operator [] (int index);

    GameState();

    GameState(int _row, int _col);

    void initialize(int _row, int _col);

    int getScore();

    // check if this operation is valid
    bool checkValid(GameOperation _o);

    // update the State, make sure _o is valid
    void updateState(GameOperation _o);

    // check if this state is an end state
    bool end();
};

#endif
