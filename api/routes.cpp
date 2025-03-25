#include <crow.h>
#include "image_processor.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void setup_routes(crow::SimpleApp& app, ImageProcessor* processor) {
    CROW_ROUTE(app, "/status").methods("GET"_method)([](){
        return crow::response(200, "API is up and running");
    });

    CROW_ROUTE(app, "/process").methods("POST"_method)([processor](const crow::request& req){
        auto body = json::parse(req.body);
        std::vector<unsigned char> input = body["image"].get<std::vector<unsigned char>>();
        int width = body["width"], height = body["height"];
        auto output = processor->process(input, width, height);
        json resp;
        resp["output"] = output;
        return crow::response(200, resp.dump());
    });
}