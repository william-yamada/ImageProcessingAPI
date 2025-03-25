#include <fstream>
#include <sstream>
#include <crow.h>
#include "image_processor.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void setup_routes(crow::SimpleApp& app, ImageProcessor* processor) {
    CROW_ROUTE(app, "/docs/<path>").methods("GET"_method)
    ([](const crow::request&, crow::response& res, std::string path){
        std::ifstream file("public/swagger/" + path, std::ios::binary);
        if (!file) {
            res.code = 404;
            res.end("File not found");
            return;
        }
        std::ostringstream contents;
        contents << file.rdbuf();
        res.set_header("Content-Type", "text/html");
        res.write(contents.str());
        res.end();
    });

    CROW_ROUTE(app, "/swagger.yaml").methods("GET"_method)
    ([](){
        std::ifstream file("api/swagger.yaml");
        if (!file) return crow::response(404, "swagger.yaml not found");
        std::ostringstream contents;
        contents << file.rdbuf();
        return crow::response{contents.str()};
    });
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
