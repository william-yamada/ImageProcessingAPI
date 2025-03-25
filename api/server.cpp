#include <crow.h>
#include "image_processor.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void setup_routes(crow::SimpleApp& app, ImageProcessor* processor);

int main() {
    crow::SimpleApp app;
    ImageProcessor* processor = create_processor();
    setup_routes(app, processor);
    app.port(18080).multithreaded().run();
    delete processor;
    return 0;
}