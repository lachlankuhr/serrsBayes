#include <functional>
#include <iostream>

using namespace std;

int main() {

    int i = 7;
    int j = 8;

    function<int (int extra)> f = [&](int extra) { return i + j + extra; };
    cout << f(700) << endl;
}