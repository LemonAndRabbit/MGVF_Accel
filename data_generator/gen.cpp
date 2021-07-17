#include <random>
#include <fstream>

int main()
{

    std::ofstream mgvf("mgvf.data");
    std::ofstream I("i.data");

    mgvf.precision(18);
    I.precision(18);
    mgvf << std::fixed;
    I << std::fixed;

    std::default_random_engine gen;
    std::uniform_int_distribution<int> dis(1,100);

    for (int i = 0; i<1024*1024; i++){
        mgvf << 0.001 * (float)dis(gen)/(float)dis(gen) << std::endl;
    }

    for (int i = 0; i<1024*1024; i++){
        I << 0.001 * (float)dis(gen)/(float)dis(gen) << std::endl;
    }

    mgvf.close();
    I.close();
    return 0;
}
