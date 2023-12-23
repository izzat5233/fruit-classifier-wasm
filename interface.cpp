#include "module.h"

#include <emscripten.h>
#include <emscripten/bind.h>

// Takes a javascript function name and calls it if it exists
#define CALL_JS_FUNC(FUNC_NAME) \
    EM_ASM({ \
        var funcName = UTF8ToString($0); \
        if (typeof window[funcName] === 'function') { \
            window[funcName](); \
        } \
    }, FUNC_NAME);


class NetworkController {
private:
    nn::vi_t dimensions;
    double alpha{};
    std::string actFunction;
    std::string lossFunction;
    nn::Module module;

    static nn::vvd_t pairToVector(const nn::vpd_t &data) {
        nn::vvd_t res;
        for (const auto &[i, j]: data) { res.push_back({i, j}); }
        return res;
    }

    static nn::act::Function stringToActivationFunction(const std::string &function) {
        if (function == "sigmoid") {
            return nn::act::sigmoid;
        } else if (function == "tanh") {
            return nn::act::tanh;
        } else {
            return nn::act::relu;
        }
    }

    static nn::loss::function_t stringToLossFunction(const std::string &function) {
        if (function == "mse") {
            return nn::loss::mse;
        } else {
            return nn::loss::sse;
        }
    }

    void _setDimensions(const nn::vi_t &networkDimensions) {
        this->dimensions = networkDimensions;
        CALL_JS_FUNC("onDimensionsSet")
    }

    void _setLearningRate(double learningRate) {
        this->alpha = learningRate;
        CALL_JS_FUNC("onLearningRateSet")
    }

    void _setActivationFunction(const std::string &function) {
        this->actFunction = function;
        CALL_JS_FUNC("onActivationFunctionSet")
    }

    void _setLossFunction(const std::string &function) {
        this->lossFunction = function;
        CALL_JS_FUNC("onLossFunctionSet")
    }

public:
    /**
     * On Construction, no events are triggered.
     */
    NetworkController() = default;

    /**
     * On initialization, events are triggered sequentially:
     * - `onDimensionsSet`
     * - `onLearningRateSet`
     * - `onActivationFunctionSet`
     * - `onLossFunctionSet`
     * - `onNetworkBuilt`
     */
    void init() {
        _setDimensions({4, 3, 4});
        _setLearningRate(0.01);
        _setActivationFunction("tanh");
        _setLossFunction("sse");
        build();
    }

    /**
     * Builds the module and triggers `onNetworkBuilt` event.
     */
    void build() {
        nn::act::Function act = stringToActivationFunction(actFunction);
        nn::loss::function_t loss = stringToLossFunction(lossFunction);
        module.setNetwork(nn::make::network(dimensions, act, loss));
        module.setLearningRate(alpha);
        CALL_JS_FUNC("onNetworkBuilt")
    }

    void setDimensions(const nn::vi_t &networkDimensions) {
        _setDimensions(networkDimensions);
        build();
    }

    [[nodiscard]] nn::vi_t getDimensions() const {
        return dimensions;
    }

    void setLearningRate(double learningRate) {
        _setLearningRate(learningRate);
        module.setLearningRate(alpha);
    }

    [[nodiscard]] double getLearningRate() const {
        return alpha;
    }

    void setActivationFunction(const std::string &function) {
        _setActivationFunction(function);
        build();
    }

    [[nodiscard]] std::string getActivationFunction() const {
        return actFunction;
    }

    void setLossFunction(const std::string &function) {
        _setLossFunction(function);
        build();
    }

    [[nodiscard]] std::string getLossFunction() const {
        return lossFunction;
    }

    [[nodiscard]] nn::vvvd_t getWeights() const {
        return module.getWeights();
    }

    [[nodiscard]] nn::vvd_t getBiases() const {
        return module.getBiases();
    }

    void setTrainInput(const nn::vvd_t &data) {
        module.setTrainInput(data);
        CALL_JS_FUNC("onTrainInputSet")
    }

    [[nodiscard]] nn::vvd_t getTrainInput() const {
        return module.getTrainInput();
    }

    void clearTrainInput() {
        module.setTrainInput({});
        CALL_JS_FUNC("onTrainInputCleared")
    }

    void setTrainOutput(const nn::vvd_t &data) {
        module.setTrainOutput(data);
        CALL_JS_FUNC("onTrainOutputSet")
    }

    [[nodiscard]] nn::vvd_t getTrainOutput() const {
        return module.getTrainOutput();
    }

    void clearTrainOutput() {
        module.setTrainOutput({});
        CALL_JS_FUNC("onTrainOutputCleared")
    }

    void setTestInput(const nn::vvd_t &data) {
        module.setTestInput(data);
        CALL_JS_FUNC("onTestInputSet")
    }

    [[nodiscard]] nn::vvd_t getTestInput() const {
        return module.getTestInput();
    }

    void clearTestInput() {
        module.setTestInput({});
        CALL_JS_FUNC("onTestInputCleared")
    }

    void setTestOutput(const nn::vvd_t &data) {
        module.setTestOutput(data);
        CALL_JS_FUNC("onTestOutputSet")
    }

    [[nodiscard]] nn::vvd_t getTestOutput() const {
        return module.getTestOutput();
    }

    void clearTestOutput() {
        module.setTestOutput({});
        CALL_JS_FUNC("onTestOutputCleared")
    }

    nn::vd_t trainFor(std::size_t epochs) {
        return module.train(epochs);
    }

    nn::vvd_t trainAndTestFor(std::size_t epochs) {
        return pairToVector(module.trainAndTest(epochs));
    }

    [[nodiscard]] nn::vvd_t getPredictions() const {
        return module.predict();
    }

    [[nodiscard]] nn::vvd_t getCustomPredictions(const nn::vvd_t &data) const {
        return module.predict(data);
    }
};

EMSCRIPTEN_BINDINGS(my_module) {
    using namespace emscripten;

    register_vector<int>("VecInt");
    register_vector<nn::ui_t>("VecUInt");
    register_vector<double>("VecNum");
    register_vector<nn::vd_t>("VecVecNum");
    register_vector<nn::vvd_t>("VecVecVecNum");

    class_<NetworkController>("Network")
            .constructor<>()
            .function("init", &NetworkController::init)
            .function("build", &NetworkController::build)
            .function("setDimensions", &NetworkController::setDimensions)
            .function("getDimensions", &NetworkController::getDimensions)
            .function("setLearningRate", &NetworkController::setLearningRate)
            .function("getLearningRate", &NetworkController::getLearningRate)
            .function("setActivationFunction", &NetworkController::setActivationFunction)
            .function("getActivationFunction", &NetworkController::getActivationFunction)
            .function("setLossFunction", &NetworkController::setLossFunction)
            .function("getLossFunction", &NetworkController::getLossFunction)
            .function("getWeights", &NetworkController::getWeights)
            .function("getBiases", &NetworkController::getBiases)
            .function("setTrainInput", &NetworkController::setTrainInput)
            .function("getTrainInput", &NetworkController::getTrainInput)
            .function("clearTrainInput", &NetworkController::clearTrainInput)
            .function("setTrainOutput", &NetworkController::setTrainOutput)
            .function("getTrainOutput", &NetworkController::getTrainOutput)
            .function("clearTrainOutput", &NetworkController::clearTrainOutput)
            .function("setTestInput", &NetworkController::setTestInput)
            .function("getTestInput", &NetworkController::getTestInput)
            .function("clearTestInput", &NetworkController::clearTestInput)
            .function("setTestOutput", &NetworkController::setTestOutput)
            .function("getTestOutput", &NetworkController::getTestOutput)
            .function("clearTestOutput", &NetworkController::clearTestOutput)
            .function("trainFor", &NetworkController::trainFor)
            .function("trainAndTestFor", &NetworkController::trainAndTestFor)
            .function("getPredictions", &NetworkController::getPredictions)
            .function("getCustomPredictions", &NetworkController::getCustomPredictions);
}

int main() {}