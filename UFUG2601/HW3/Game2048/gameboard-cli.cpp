#include "gameboard-cli.h"
#include "configuration.h"

enum ViewState {
    Playing, RankList, Welcome, Pause
} currentView;
bool keepGoing;
Configuration configuration;

void updateView_Playing() {

}
void updateView_RankList() {

}

void updateView_Welcome() {

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

void inputHandler() {
    
}
void exec2048() {
    configuration.load("default.config");
    keepGoing = true;
    while (keepGoing) {
        updateView();
        inputHandler();
    }
    configuration.save();
}