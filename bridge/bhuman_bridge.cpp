/**
 * @file bhuman_bridge.cpp
 *
 * Standalone BHuman C++ bridge for RL integration.
 *
 * This program creates a headless BHuman Robot in 2D mode,
 * accepts GroundTruthWorldState frames as JSON on stdin,
 * runs one cognition cycle per frame, and returns decisions
 * (SkillRequest, MotionRequest, ActivationGraph, MotionInfo)
 * as JSON on stdout.
 *
 * Protocol: JSON-line over stdin/stdout (see bridge_protocol.py).
 *
 * Build: cmake --build . --config Release
 *        (requires BHuman source tree at BHUMAN_ROOT)
 *
 * Architecture:
 *   main() -> BridgeRunner -> Robot (BHuman framework)
 *                           -> DataInjector (replaces LocalConsole)
 *                           -> ResultReader (reads Blackboard)
 *
 * Data flow (mirrors LocalConsole::main() for 2D):
 *   JSON stdin -> parse -> GroundTruthWorldState
 *              -> inject via debugSender:
 *                   idFrameBegin "Cognition"
 *                   idFrameInfo
 *                   idCameraInfo
 *                   idGroundTruthOdometryData
 *                   idFallDownState (upright)
 *                   idGroundContactState (contact=true)
 *                   idCameraMatrix (isValid=false)
 *                   idMotionInfo
 *                   idGameControllerData
 *                   idGroundTruthWorldState
 *                   idFrameFinished "Cognition"
 *              -> Cognition thread processes
 *              -> read SkillRequest, MotionRequest, ActivationGraph
 *              -> serialize to JSON -> stdout
 *
 * @author SabanaHerons2026 RL Integration
 */

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <condition_variable>
#ifdef WINDOWS
#include <Windows.h>
#include <timeapi.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// BHuman framework includes
// ─────────────────────────────────────────────────────────────────────────────
#include "Framework/Robot.h"
#include "Framework/Settings.h"
#include "Framework/Debug.h"
#include "Framework/ModuleContainer.h"
#include "Framework/ModuleGraphRunner.h"
#include "Framework/Blackboard.h"
#include "Framework/ThreadFrame.h"
#include "Framework/Communication.h"
#include "Platform/SystemCall.h"
#include "Platform/Time.h"

#include "Debugging/DebugRequest.h"
#include "Streaming/MessageIDs.h"
#include "Streaming/FunctionList.h"
#include "Streaming/InStreams.h"
#include "Streaming/OutStreams.h"

// Representations we inject
#include "Representations/Infrastructure/GroundTruthWorldState.h"
#include "Representations/Infrastructure/FrameInfo.h"
#include "Representations/Infrastructure/CameraInfo.h"
#include "Representations/Communication/GameControllerData.h"
#include "Representations/MotionControl/MotionInfo.h"
#include "Representations/MotionControl/OdometryData.h"
#include "Representations/Infrastructure/SensorData/FsrSensorData.h"
#include "Representations/Infrastructure/SensorData/InertialSensorData.h"
#include "Representations/Infrastructure/SensorData/JointSensorData.h"
#include "Representations/Sensing/FallDownState.h"
#include "Representations/Sensing/GroundContactState.h"
#include "Representations/Perception/ImagePreprocessing/CameraMatrix.h"

// Representations we read
#include "Representations/BehaviorControl/SkillRequest.h"
#include "Representations/BehaviorControl/ActivationGraph.h"
#include "Representations/BehaviorControl/BehaviorStatus.h"
#include "Representations/BehaviorControl/StrategyStatus.h"
#include "Representations/MotionControl/MotionRequest.h"
#include "Representations/Infrastructure/GameState.h"
#include "Streaming/TypeRegistry.h"
#include "Tools/BehaviorControl/Strategy/Role.h"

SystemCall::Mode SystemCall::getMode()
{
    return SystemCall::simulatedRobot;
}

#ifdef WINDOWS
static LONG WINAPI rlBridgeUnhandledExceptionFilter(EXCEPTION_POINTERS* exceptionInfo)
{
    if(exceptionInfo && exceptionInfo->ExceptionRecord)
    {
        std::cerr << "[bridge] CRASH! Exception code: 0x"
                  << std::hex << exceptionInfo->ExceptionRecord->ExceptionCode
                  << " at address: 0x"
                  << reinterpret_cast<uintptr_t>(exceptionInfo->ExceptionRecord->ExceptionAddress)
                  << std::dec << std::endl;
    }
    else
        std::cerr << "[bridge] CRASH! Unknown unhandled exception" << std::endl;
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

namespace
{
constexpr bool kVerboseBridgeLogs = false;

template<typename... Ts>
void verboseBridgeLog(const Ts&... parts)
{
    if constexpr(kVerboseBridgeLogs)
    {
        (std::cerr << ... << parts);
        std::cerr << std::endl;
    }
}
}


// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON parser/writer (no external dependency)
// Uses simple recursive descent for parsing and string concatenation for output.
// Production code should use nlohmann/json or similar.
// ─────────────────────────────────────────────────────────────────────────────

#include <sstream>
#include <map>
#include <variant>
#include <optional>

// Forward declarations for JSON types
struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue
{
    std::variant<
        std::nullptr_t,
        bool,
        double,
        std::string,
        JsonArray,
        JsonObject
    > data;

    JsonValue() : data(nullptr) {}
    JsonValue(bool b) : data(b) {}
    JsonValue(int i) : data(static_cast<double>(i)) {}
    JsonValue(double d) : data(d) {}
    JsonValue(const std::string& s) : data(s) {}
    JsonValue(const char* s) : data(std::string(s)) {}
    JsonValue(const JsonArray& a) : data(a) {}
    JsonValue(const JsonObject& o) : data(o) {}

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool isBool() const { return std::holds_alternative<bool>(data); }
    bool isNumber() const { return std::holds_alternative<double>(data); }
    bool isString() const { return std::holds_alternative<std::string>(data); }
    bool isArray() const { return std::holds_alternative<JsonArray>(data); }
    bool isObject() const { return std::holds_alternative<JsonObject>(data); }

    double asNumber(double def = 0.0) const { return isNumber() ? std::get<double>(data) : def; }
    int asInt(int def = 0) const { return isNumber() ? static_cast<int>(std::get<double>(data)) : def; }
    bool asBool(bool def = false) const { return isBool() ? std::get<bool>(data) : def; }
    const std::string& asString() const { static std::string empty; return isString() ? std::get<std::string>(data) : empty; }
    const JsonArray& asArray() const { static JsonArray empty; return isArray() ? std::get<JsonArray>(data) : empty; }
    const JsonObject& asObject() const { static JsonObject empty; return isObject() ? std::get<JsonObject>(data) : empty; }

    const JsonValue& operator[](const std::string& key) const
    {
        static JsonValue null;
        if (!isObject()) return null;
        auto it = std::get<JsonObject>(data).find(key);
        return it != std::get<JsonObject>(data).end() ? it->second : null;
    }

    const JsonValue& operator[](size_t idx) const
    {
        static JsonValue null;
        if (!isArray()) return null;
        auto& arr = std::get<JsonArray>(data);
        return idx < arr.size() ? arr[idx] : null;
    }
};

// Minimal JSON parser
class JsonParser
{
public:
    static std::optional<JsonValue> parse(const std::string& input)
    {
        JsonParser p(input);
        p.skipWhitespace();
        auto val = p.parseValue();
        return val;
    }

private:
    const std::string& src;
    size_t pos = 0;

    JsonParser(const std::string& s) : src(s) {}

    void skipWhitespace()
    {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r'))
            pos++;
    }

    char peek() { return pos < src.size() ? src[pos] : '\0'; }
    char advance() { return pos < src.size() ? src[pos++] : '\0'; }

    std::optional<JsonValue> parseValue()
    {
        skipWhitespace();
        char c = peek();
        if (c == '"') return parseString();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        return std::nullopt;
    }

    std::optional<JsonValue> parseString()
    {
        advance(); // skip "
        std::string result;
        while (pos < src.size() && src[pos] != '"')
        {
            if (src[pos] == '\\')
            {
                pos++;
                if (pos < src.size())
                {
                    switch (src[pos])
                    {
                        case '"': result += '"'; break;
                        case '\\': result += '\\'; break;
                        case '/': result += '/'; break;
                        case 'n': result += '\n'; break;
                        case 't': result += '\t'; break;
                        default: result += src[pos]; break;
                    }
                }
            }
            else
                result += src[pos];
            pos++;
        }
        advance(); // skip closing "
        return JsonValue(result);
    }

    std::optional<JsonValue> parseNumber()
    {
        size_t start = pos;
        if (peek() == '-') advance();
        while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') advance();
        if (pos < src.size() && src[pos] == '.')
        {
            advance();
            while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') advance();
        }
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E'))
        {
            advance();
            if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) advance();
            while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') advance();
        }
        return JsonValue(std::stod(src.substr(start, pos - start)));
    }

    std::optional<JsonValue> parseBool()
    {
        if (src.substr(pos, 4) == "true") { pos += 4; return JsonValue(true); }
        if (src.substr(pos, 5) == "false") { pos += 5; return JsonValue(false); }
        return std::nullopt;
    }

    std::optional<JsonValue> parseNull()
    {
        if (src.substr(pos, 4) == "null") { pos += 4; return JsonValue(); }
        return std::nullopt;
    }

    std::optional<JsonValue> parseArray()
    {
        advance(); // skip [
        JsonArray arr;
        skipWhitespace();
        if (peek() == ']') { advance(); return JsonValue(arr); }
        while (true)
        {
            auto val = parseValue();
            if (!val) return std::nullopt;
            arr.push_back(*val);
            skipWhitespace();
            if (peek() == ',') { advance(); skipWhitespace(); }
            else break;
        }
        if (peek() == ']') advance();
        return JsonValue(arr);
    }

    std::optional<JsonValue> parseObject()
    {
        advance(); // skip {
        JsonObject obj;
        skipWhitespace();
        if (peek() == '}') { advance(); return JsonValue(obj); }
        while (true)
        {
            skipWhitespace();
            auto key = parseString();
            if (!key || !key->isString()) return std::nullopt;
            skipWhitespace();
            advance(); // skip :
            auto val = parseValue();
            if (!val) return std::nullopt;
            obj[key->asString()] = *val;
            skipWhitespace();
            if (peek() == ',') { advance(); }
            else break;
        }
        skipWhitespace();
        if (peek() == '}') advance();
        return JsonValue(obj);
    }
};

// Minimal JSON serializer
std::string jsonSerialize(const JsonValue& val)
{
    if (val.isNull()) return "null";
    if (val.isBool()) return val.asBool() ? "true" : "false";
    if (val.isNumber())
    {
        double d = val.asNumber();
        if (d == static_cast<int>(d) && std::abs(d) < 1e15)
        {
            return std::to_string(static_cast<long long>(d));
        }
        std::ostringstream oss;
        oss << d;
        return oss.str();
    }
    if (val.isString())
    {
        std::string result = "\"";
        for (char c : val.asString())
        {
            switch (c)
            {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\t': result += "\\t"; break;
                default: result += c;
            }
        }
        result += "\"";
        return result;
    }
    if (val.isArray())
    {
        std::string result = "[";
        bool first = true;
        for (auto& item : val.asArray())
        {
            if (!first) result += ",";
            result += jsonSerialize(item);
            first = false;
        }
        result += "]";
        return result;
    }
    if (val.isObject())
    {
        std::string result = "{";
        bool first = true;
        for (auto& [key, v] : val.asObject())
        {
            if (!first) result += ",";
            result += "\"" + key + "\":" + jsonSerialize(v);
            first = false;
        }
        result += "}";
        return result;
    }
    return "null";
}


// ─────────────────────────────────────────────────────────────────────────────
// BRIDGE RUNNER — core bridge logic
// ─────────────────────────────────────────────────────────────────────────────

struct DecisionSnapshot
{
    unsigned completedCognitionFrames = 0;
    bool hasGameState = false;
    bool hasSkillRequest = false;
    bool hasMotionRequest = false;
    bool hasMotionInfo = false;
    bool hasStrategyStatus = false;
    bool hasBehaviorStatus = false;
    bool hasActivationGraph = false;

    GameState gameState;
    SkillRequest skillRequest;
    MotionRequest motionRequest;
    MotionInfo motionInfo;
    StrategyStatus strategyStatus;
    BehaviorStatus behaviorStatus;
    ActivationGraph activationGraph;
};

class RLBridgeConsole : public ThreadFrame
{
public:
    RLBridgeConsole(const Settings& settings, const std::string& robotName, Debug* debug);

    DebugSender<MessageQueue>& sender();
    void clearSnapshots();
    unsigned cognitionFrameFinishedCount() const;
    unsigned motionFrameFinishedCount() const;
    bool waitForCognitionFrameFinished(unsigned previousCount, int timeoutMs);
    bool waitForMotionFrameFinished(unsigned previousCount, int timeoutMs);
    DecisionSnapshot latestSnapshot() const;
    unsigned totalReceivedMessages() const;
    std::vector<std::string> recentReceivedMessages() const;

protected:
    int getPriority() const override { return 0; }
    void init() override {}
    bool main() override { return true; }
    void terminate() override {}
    bool handleMessage(MessageQueue::Message message) override;

private:
    DebugReceiver<MessageQueue>* connectReceiverWithRobot(Debug* debug);
    DebugSender<MessageQueue>* connectSenderWithRobot(Debug* debug) const;

    mutable std::mutex receiverMutex_;
    std::condition_variable receiverCv_;
    unsigned cognitionFrameFinishedCount_ = 0;
    unsigned motionFrameFinishedCount_ = 0;
    DecisionSnapshot pendingSnapshot_;
    DecisionSnapshot latestSnapshot_;
    unsigned totalReceivedMessages_ = 0;
    std::deque<std::string> recentReceivedMessages_;
    bool captureDecisionSnapshot_ = false;
};

RLBridgeConsole::RLBridgeConsole(const Settings& settings, const std::string& robotName, Debug* debug) :
    ThreadFrame(settings, robotName, connectReceiverWithRobot(debug), connectSenderWithRobot(debug))
{
}

DebugSender<MessageQueue>& RLBridgeConsole::sender()
{
    return *debugSender;
}

void RLBridgeConsole::clearSnapshots()
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    cognitionFrameFinishedCount_ = 0;
    motionFrameFinishedCount_ = 0;
    pendingSnapshot_ = DecisionSnapshot();
    latestSnapshot_ = DecisionSnapshot();
    totalReceivedMessages_ = 0;
    recentReceivedMessages_.clear();
    captureDecisionSnapshot_ = false;
}

unsigned RLBridgeConsole::cognitionFrameFinishedCount() const
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    return cognitionFrameFinishedCount_;
}

unsigned RLBridgeConsole::motionFrameFinishedCount() const
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    return motionFrameFinishedCount_;
}

bool RLBridgeConsole::waitForCognitionFrameFinished(unsigned previousCount, int timeoutMs)
{
    std::unique_lock<std::mutex> lock(receiverMutex_);
    return receiverCv_.wait_for(
        lock,
        std::chrono::milliseconds(timeoutMs),
        [&] { return cognitionFrameFinishedCount_ > previousCount; });
}

bool RLBridgeConsole::waitForMotionFrameFinished(unsigned previousCount, int timeoutMs)
{
    std::unique_lock<std::mutex> lock(receiverMutex_);
    return receiverCv_.wait_for(
        lock,
        std::chrono::milliseconds(timeoutMs),
        [&] { return motionFrameFinishedCount_ > previousCount; });
}

DecisionSnapshot RLBridgeConsole::latestSnapshot() const
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    return latestSnapshot_;
}

unsigned RLBridgeConsole::totalReceivedMessages() const
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    return totalReceivedMessages_;
}

std::vector<std::string> RLBridgeConsole::recentReceivedMessages() const
{
    std::lock_guard<std::mutex> lock(receiverMutex_);
    return std::vector<std::string>(recentReceivedMessages_.begin(), recentReceivedMessages_.end());
}

bool RLBridgeConsole::handleMessage(MessageQueue::Message message)
{
    const char* messageNameForLog = "unknown_msg";
    switch(message.id())
    {
    case idFrameBegin:
        messageNameForLog = "idFrameBegin";
        break;
    case idGameState:
        messageNameForLog = "idGameState";
        break;
    case idSkillRequest:
        messageNameForLog = "idSkillRequest";
        break;
    case idMotionRequest:
        messageNameForLog = "idMotionRequest";
        break;
    case idMotionInfo:
        messageNameForLog = "idMotionInfo";
        break;
    case idStrategyStatus:
        messageNameForLog = "idStrategyStatus";
        break;
    case idBehaviorStatus:
        messageNameForLog = "idBehaviorStatus";
        break;
    case idActivationGraph:
        messageNameForLog = "idActivationGraph";
        break;
    case idLogResponse:
        messageNameForLog = "idLogResponse";
        break;
    case idFrameFinished:
        messageNameForLog = "idFrameFinished";
        break;
    default:
        break;
    }
    verboseBridgeLog("[bridge-console] handleMessage: ",
                     messageNameForLog,
                     " (",
                     static_cast<int>(message.id()),
                     ")");

    {
        std::lock_guard<std::mutex> lock(receiverMutex_);
        ++totalReceivedMessages_;
        recentReceivedMessages_.push_back(std::string(messageNameForLog));
        if(recentReceivedMessages_.size() > 16)
            recentReceivedMessages_.pop_front();
    }

    switch (message.id())
    {
    case idFrameBegin:
    {
        auto stream = message.bin();
        std::string threadName;
        stream >> threadName;
        {
            std::lock_guard<std::mutex> lock(receiverMutex_);
            if(threadName == "Cognition")
            {
                pendingSnapshot_ = DecisionSnapshot();
                captureDecisionSnapshot_ = true;
            }
            else
                captureDecisionSnapshot_ = false;
        }
        return true;
    }
    case idGameState:
        verboseBridgeLog("[bridge-console] deserialize GameState");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.gameState;
            pendingSnapshot_.hasGameState = true;
        }
        return true;
    case idSkillRequest:
        verboseBridgeLog("[bridge-console] deserialize SkillRequest");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.skillRequest;
            pendingSnapshot_.hasSkillRequest = true;
        }
        return true;
    case idMotionRequest:
        verboseBridgeLog("[bridge-console] deserialize MotionRequest");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.motionRequest;
            pendingSnapshot_.hasMotionRequest = true;
        }
        return true;
    case idMotionInfo:
        verboseBridgeLog("[bridge-console] deserialize MotionInfo");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.motionInfo;
            pendingSnapshot_.hasMotionInfo = true;
        }
        return true;
    case idStrategyStatus:
        verboseBridgeLog("[bridge-console] deserialize StrategyStatus");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.strategyStatus;
            pendingSnapshot_.hasStrategyStatus = true;
        }
        return true;
    case idBehaviorStatus:
        verboseBridgeLog("[bridge-console] deserialize BehaviorStatus");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.behaviorStatus;
            pendingSnapshot_.hasBehaviorStatus = true;
        }
        return true;
    case idActivationGraph:
        verboseBridgeLog("[bridge-console] deserialize ActivationGraph");
        if(captureDecisionSnapshot_)
        {
            message.bin() >> pendingSnapshot_.activationGraph;
            pendingSnapshot_.hasActivationGraph = true;
        }
        return true;
    case idLogResponse:
        verboseBridgeLog("[bridge-console] received LogResponse");
        return true;
    case idFrameFinished:
    {
        verboseBridgeLog("[bridge-console] received FrameFinished");
        auto stream = message.bin();
        std::string threadName;
        stream >> threadName;
        verboseBridgeLog("[bridge-console] FrameFinished thread=", threadName);
        if (threadName == "Cognition")
        {
            {
                std::lock_guard<std::mutex> lock(receiverMutex_);
                latestSnapshot_ = pendingSnapshot_;
                latestSnapshot_.completedCognitionFrames = ++cognitionFrameFinishedCount_;
                captureDecisionSnapshot_ = false;
            }
            receiverCv_.notify_all();
        }
        else if(threadName == "Motion")
        {
            {
                std::lock_guard<std::mutex> lock(receiverMutex_);
                ++motionFrameFinishedCount_;
            }
            receiverCv_.notify_all();
        }
        return true;
    }
    default:
        return ThreadFrame::handleMessage(message);
    }
}

DebugReceiver<MessageQueue>* RLBridgeConsole::connectReceiverWithRobot(Debug* debug)
{
    ASSERT(!debug->debugSender);
    auto* receiver = new DebugReceiver<MessageQueue>(this, debug->getName());
    debug->debugSender = new DebugSender<MessageQueue>(*receiver, "RLBridgeConsole");
    return receiver;
}

DebugSender<MessageQueue>* RLBridgeConsole::connectSenderWithRobot(Debug* debug) const
{
    ASSERT(!debug->debugReceiver);
    debug->debugReceiver = new DebugReceiver<MessageQueue>(debug, "RLBridgeConsole");
    return new DebugSender<MessageQueue>(*debug->debugReceiver, debug->getName());
}

class RLBridgeRobot : public Robot
{
public:
    RLBridgeRobot(const Settings& settings, const std::string& name) :
        Robot(settings, name),
        console(new RLBridgeConsole(settings, name, static_cast<Debug*>(front())))
    {
        push_back(console);
    }

    RLBridgeConsole* getConsole() const
    {
        return console;
    }

private:
    RLBridgeConsole* console;
};

struct DebugSenderBaseAccess : public DebugSenderBase
{
    static void setTerminating(bool value)
    {
        terminating = value;
    }
};

class BridgeRunner
{
public:
    BridgeRunner() = default;
    ~BridgeRunner() { shutdown(); }

    /**
     * Initialize the BHuman Robot with given settings.
     * Creates the Robot instance, connects our data injector to the Debug thread,
     * and starts all threads.
     */
    bool init(const std::string& scenario, const std::string& location,
              int teamNumber, int playerNumber, int numPlayers,
              const std::string& bridgeMode)
    {
        try
        {
            if (!functionListExecuted_)
            {
                FunctionList::execute();
                functionListExecuted_ = true;
            }

            if (!timeInitialized_)
            {
                Time::initialize();
                Time::setSimulatedTime(true);
                timeInitialized_ = true;
            }
#ifdef WINDOWS
            if (!timerResolutionSet_)
            {
                VERIFY(timeBeginPeriod(1) == TIMERR_NOERROR);
                timerResolutionSet_ = true;
            }
#endif
            DebugSenderBaseAccess::setTerminating(false);

            teamNumber_ = teamNumber;
            playerNumber_ = playerNumber;
            numPlayers_ = numPlayers;
            bridgeMode_ = bridgeMode.empty() ? "persistent_2d_cognition" : bridgeMode;

            // Create Settings for the specified scenario
            // The search path will include Config/Scenarios/{scenario}/ which
            // determines threads.cfg (Cognition2D for "2D")
            settings_ = std::make_unique<Settings>(
                "Nao",            // headName
                "Nao",            // bodyName
                teamNumber,       // teamNumber
                Settings::blue,   // fieldPlayerColor
                Settings::blue,   // goalkeeperColor
                playerNumber,     // playerNumber
                location,         // location
                scenario,         // scenario (e.g. "2D")
                0                 // magicNumber
            );

            // Create a Robot with a real ThreadFrame-based bridge console.
            // This mirrors PythonConsole/LocalConsole ownership and avoids
            // wiring Debug sender/receiver from a non-thread object.
            robot_ = std::make_unique<RLBridgeRobot>(*settings_, "RLBridge");
            std::cerr << "[bridge] init: robot created" << std::endl;
            bridgeConsole_ = robot_->getConsole();
            if(!bridgeConsole_)
                throw std::runtime_error("RLBridgeConsole was not created");

            // Start all robot threads
            robot_->start();
            std::cerr << "[bridge] init: robot started" << std::endl;

            // Give threads time to initialize
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            cognition_ = findCognitionThread();
            if (!cognition_)
                throw std::runtime_error("Cognition thread not found");
            std::cerr << "[bridge] init: cognition found" << std::endl;

            motion_ = findMotionThread();
            if(motion_)
                std::cerr << "[bridge] init: motion found" << std::endl;
            else
                std::cerr << "[bridge] init: motion not found, motion evidence disabled" << std::endl;

            bridgeConsole_->clearSnapshots();
            sendDebugRequest("module:GameStateProvider:active");
            std::cerr << "[bridge] init: requested GameStateProvider active" << std::endl;
            std::cerr << "[bridge] init: using debug-channel ack plus direct blackboard fallback" << std::endl;

            frameId_ = 0;
            lastInputTimestampMs_ = -1;
            initialized_ = true;

            std::cerr << "[bridge] Robot initialized with scenario=" << scenario
                      << " location=" << location
                      << " bridge_mode=" << bridgeMode_
                      << " player=" << playerNumber << std::endl;
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[bridge] Init failed: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * Process one frame: inject world state, run cognition, read decisions.
     *
     * This replicates what LocalConsole::main() does for 2D mode:
     *   1. Build BHuman representations from JSON input
     *   2. Send them via debugSender in the same order as LocalConsole
     *   3. Wait for cognition to process (timing-based for now)
     *   4. Read results from the Cognition thread's Blackboard
     */
    JsonValue processFrame(const JsonValue& input)
    {
        if (!initialized_)
            return makeError("Bridge not initialized");

        auto start = std::chrono::high_resolution_clock::now();
        verboseBridgeLog("[bridge] processFrame: begin");

        frameId_++;
        const int baseInputTimestampMs = input["timestamp_ms"].asInt(frameId_ * 33);
        const std::string requestedState = input["game_state"].asString().empty() ? "playing" : input["game_state"].asString();
        const std::vector<std::string> bootstrapStates = determineBootstrapStates(requestedState);

        GroundTruthWorldState worldState;
        buildWorldState(input, worldState);
        verboseBridgeLog("[bridge] processFrame: world built");

        CameraInfo cameraInfo;
        fillDefaultCameraInfo(cameraInfo);

        GroundTruthOdometryData odometryData;

        FallDownState fallDownState;
        fallDownState.state = FallDownState::upright;

        GroundContactState groundContactState;
        groundContactState.contact = true;

        CameraMatrix cameraMatrix;
        cameraMatrix.isValid = false;

        unsigned finalBridgeTimestamp = 0;
        bool frameCompleted = true;
        bool motionCompleted = false;
        for(size_t stepIndex = 0; stepIndex < bootstrapStates.size(); ++stepIndex)
        {
            const unsigned bridgeTimestamp = advanceBridgeTime(baseInputTimestampMs + static_cast<int>(stepIndex * 33));
            finalBridgeTimestamp = bridgeTimestamp;

            FrameInfo frameInfo;
            frameInfo.time = bridgeTimestamp;

            GameControllerData gameControllerData;
            buildGameControllerData(input, bridgeTimestamp, gameControllerData, bootstrapStates[stepIndex]);

            // Feed Cognition with an operational motion state instead of the
            // MotionInfo default (playDead). Leaving the default here keeps the
            // behavior chain in a non-executable posture and prevents
            // MotionRequest / kick evidence from surfacing downstream.
            MotionInfo motionInfo;
            motionInfo.executedPhase = MotionPhase::stand;
            motionInfo.isMotionStable = true;
            motionInfo.speed = Pose2f();
            motionInfo.lastStandTimeStamp = bridgeTimestamp;
            motionInfo.lastMotionPhaseStarted = bridgeTimestamp;

            JointSensorData jointSensorData;
            jointSensorData.timestamp = bridgeTimestamp;
            FsrSensorData fsrSensorData;
            InertialSensorData inertialSensorData;

            const unsigned previousFinishedCount = bridgeConsole_->cognitionFrameFinishedCount();
            auto& bridgeSender = bridgeConsole_->sender();
            bridgeSender.bin(idFrameBegin) << std::string("Cognition");
            bridgeSender.bin(idFrameInfo) << frameInfo;
            bridgeSender.bin(idCameraInfo) << cameraInfo;
            bridgeSender.bin(idGroundTruthOdometryData) << odometryData;
            bridgeSender.bin(idFallDownState) << fallDownState;
            bridgeSender.bin(idGroundContactState) << groundContactState;
            bridgeSender.bin(idCameraMatrix) << cameraMatrix;
            bridgeSender.bin(idMotionInfo) << motionInfo;
            bridgeSender.bin(idGameControllerData) << gameControllerData;
            bridgeSender.bin(idGroundTruthWorldState) << worldState;
            bridgeSender.bin(idFrameFinished) << std::string("Cognition");
            bridgeSender.send(true);
            verboseBridgeLog("[bridge] processFrame: frame sent state=",
                             bootstrapStates[stepIndex],
                             " step=",
                             (stepIndex + 1),
                             "/",
                             bootstrapStates.size());

            frameCompleted = bridgeConsole_->waitForCognitionFrameFinished(previousFinishedCount, 1000);
            if(frameCompleted)
                verboseBridgeLog("[bridge] processFrame: cognition frame finished state=",
                                 bootstrapStates[stepIndex]);
            else
            {
                verboseBridgeLog("[bridge] processFrame: frame completion timeout for state=",
                                 bootstrapStates[stepIndex],
                                 ", using snapshot fallback");
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }

            if(motion_)
            {
                const unsigned previousMotionFinishedCount = bridgeConsole_->motionFrameFinishedCount();
                auto& motionSender = bridgeConsole_->sender();
                motionSender.bin(idFrameBegin) << std::string("Motion");
                motionSender.bin(idJointSensorData) << jointSensorData;
                motionSender.bin(idFsrSensorData) << fsrSensorData;
                motionSender.bin(idInertialSensorData) << inertialSensorData;
                motionSender.bin(idGroundTruthOdometryData) << odometryData;
                motionSender.bin(idFrameFinished) << std::string("Motion");
                motionSender.send(true);
                verboseBridgeLog("[bridge] processFrame: motion frame sent state=",
                                 bootstrapStates[stepIndex]);

                motionCompleted = bridgeConsole_->waitForMotionFrameFinished(previousMotionFinishedCount, 1000);
                if(motionCompleted)
                    verboseBridgeLog("[bridge] processFrame: motion frame finished state=",
                                     bootstrapStates[stepIndex]);
                else
                    verboseBridgeLog("[bridge] processFrame: motion frame timeout state=",
                                     bootstrapStates[stepIndex]);
            }
        }

        if(requestedState == "ready" || requestedState == "set" || requestedState == "playing")
            operationalWarmupDone_ = true;

        JsonValue result = readDecisions(input["frame_id"].asInt(frameId_));
        verboseBridgeLog("[bridge] processFrame: decisions read");

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (result.isObject())
        {
            auto& obj = const_cast<JsonObject&>(result.asObject());
            obj["decision_time_us"] = JsonValue(static_cast<int>(duration.count()));
            obj["debug_frame_sync"] = JsonValue(frameCompleted ? "ack" : "timeout_fallback");
            obj["debug_motion_sync"] = JsonValue(!motion_ ? "motion_unavailable" : (motionCompleted ? "ack" : "timeout_fallback"));
            obj["debug_bootstrap_steps"] = JsonValue(static_cast<int>(bootstrapStates.size()));
            obj["debug_requested_state"] = JsonValue(requestedState);
            obj["debug_final_bridge_time"] = JsonValue(static_cast<int>(finalBridgeTimestamp));
        }

        return result;
    }


    void reset()
    {
        frameId_ = 0;
        lastInputTimestampMs_ = -1;
        operationalWarmupDone_ = false;
        if (bridgeConsole_)
            bridgeConsole_->clearSnapshots();
        if(initialized_ && bridgeConsole_)
        {
            sendDebugRequest("module:GameStateProvider:active");
            std::cerr << "[bridge] reset: requested GameStateProvider active" << std::endl;
        }
        // TODO: Reset BHuman internal state for new episode
        // This may involve sending a specific game state transition
        std::cerr << "[bridge] Reset" << std::endl;
    }

    void shutdown()
    {
        if (!robot_ && !timeInitialized_ && !timerResolutionSet_)
        {
            bridgeConsole_ = nullptr;
            cognition_ = nullptr;
            initialized_ = false;
            return;
        }

        std::cerr << "[bridge] shutdown: begin" << std::endl;
        DebugSenderBaseAccess::setTerminating(true);
        std::cerr << "[bridge] shutdown: terminating set" << std::endl;

        if (robot_)
        {
            std::cerr << "[bridge] shutdown: announceStop" << std::endl;
            robot_->announceStop();
            std::cerr << "[bridge] shutdown: stop" << std::endl;
            robot_->stop();
            std::cerr << "[bridge] shutdown: reset robot" << std::endl;
            robot_.reset();
            std::cerr << "[bridge] shutdown: robot reset done" << std::endl;
        }
        bridgeConsole_ = nullptr;
        cognition_ = nullptr;
        motion_ = nullptr;
        initialized_ = false;
        if (timeInitialized_)
        {
            std::cerr << "[bridge] shutdown: deinitialize time" << std::endl;
            Time::deinitialize();
            timeInitialized_ = false;
            std::cerr << "[bridge] shutdown: time deinitialized" << std::endl;
        }
#ifdef WINDOWS
        if (timerResolutionSet_)
        {
            std::cerr << "[bridge] shutdown: timeEndPeriod" << std::endl;
            VERIFY(timeEndPeriod(1) == TIMERR_NOERROR);
            timerResolutionSet_ = false;
            std::cerr << "[bridge] shutdown: timeEndPeriod done" << std::endl;
        }
#endif
        std::cerr << "[bridge] Shutdown" << std::endl;
    }

private:
    std::unique_ptr<Settings> settings_;
    std::unique_ptr<RLBridgeRobot> robot_;
    RLBridgeConsole* bridgeConsole_ = nullptr;
    ModuleContainer* cognition_ = nullptr;
    ModuleContainer* motion_ = nullptr;
    int teamNumber_ = 1;
    int playerNumber_ = 1;
    int numPlayers_ = 4;
    std::string bridgeMode_ = "persistent_2d_cognition";
    int frameId_ = 0;
    int lastInputTimestampMs_ = -1;
    bool operationalWarmupDone_ = false;
    bool initialized_ = false;
    bool timeInitialized_ = false;
    bool timerResolutionSet_ = false;
    bool functionListExecuted_ = false;

    void sendDebugRequest(const std::string& name, bool enable = true)
    {
        ASSERT(bridgeConsole_);
        auto& bridgeSender = bridgeConsole_->sender();
        bridgeSender.bin(idDebugRequest) << DebugRequest(name, enable);
        bridgeSender.send(true);
    }

    unsigned advanceBridgeTime(int inputTimestampMs)
    {
        int deltaMs = 33;

        if(lastInputTimestampMs_ >= 0)
            deltaMs = inputTimestampMs - lastInputTimestampMs_;
        else if(inputTimestampMs > 0)
            deltaMs = inputTimestampMs;

        // Keep the simulated BHuman clock monotonic and in a sane range.
        if(deltaMs <= 0)
            deltaMs = 33;
        else if(deltaMs > 1000)
            deltaMs = 1000;

        Time::addSimulatedTime(deltaMs);
        lastInputTimestampMs_ = inputTimestampMs;
        return Time::getCurrentSystemTime();
    }

    std::vector<std::string> determineBootstrapStates(const std::string& requestedState) const
    {
        if(operationalWarmupDone_)
            return {requestedState};

        if(requestedState == "initial" || requestedState == "finished")
            return {requestedState};
        if(requestedState == "ready")
            return {"initial", "ready"};
        if(requestedState == "set")
            return {"initial", "ready", "set"};
        if(requestedState == "playing")
            return {"initial", "ready", "set", "playing"};
        return {requestedState};
    }

    static std::string enumNameOr(const char* name, const std::string& fallback)
    {
        return name ? std::string(name) : fallback;
    }

    template<typename EnumType> static std::string enumNameOr(EnumType value, const std::string& fallback)
    {
        return enumNameOr(TypeRegistry::getEnumName(value), fallback);
    }

    static std::string normalizeTacticName(std::string name)
    {
        if(name == "t121")
            return "t121_4v4_scaled";
        if(name == "t13")
            return "t13_4v4_scaled";
        return name;
    }

    ModuleContainer* findCognitionThread() const
    {
        if (!robot_)
            return nullptr;

        for (ThreadFrame* thread : *robot_)
        {
            ModuleContainer* container = dynamic_cast<ModuleContainer*>(thread);
            if (container && thread->getName() == "Cognition")
                return container;
        }
        return nullptr;
    }

    ModuleContainer* findMotionThread() const
    {
        if(!robot_)
            return nullptr;

        for(ThreadFrame* thread : *robot_)
        {
            ModuleContainer* container = dynamic_cast<ModuleContainer*>(thread);
            if(container && thread->getName() == "Motion")
                return container;
        }
        return nullptr;
    }

    void fillDefaultCameraInfo(CameraInfo& cameraInfo) const
    {
        cameraInfo.camera = CameraInfo::upper;
        cameraInfo.width = 640;
        cameraInfo.height = 480;
        cameraInfo.openingAngleWidth = 1.0472f;
        cameraInfo.openingAngleHeight = 0.785398f;
        cameraInfo.opticalCenter = Vector2f(320.f, 240.f);
        cameraInfo.updateFocalLength();
    }

    /**
     * Build GroundTruthWorldState from JSON input.
     *
     * JSON format:
     * {
     *   "ball": {"x": 100.0, "y": 200.0, "vx": 0.0, "vy": 0.0},
     *   "own_team": [{"number": 1, "x": -2000, "y": 0, "theta": 0, "upright": true}, ...],
     *   "opp_team": [{"number": 1, "x": 2000, "y": 0, "theta": 3.14, "upright": true}, ...]
     * }
     */
    void buildWorldState(const JsonValue& input, GroundTruthWorldState& ws)
    {
        // Ball
        const auto& ball = input["ball"];
        GroundTruthWorldState::GroundTruthBall gtBall;
        gtBall.position = Vector3f(
            static_cast<float>(ball["x"].asNumber()),
            static_cast<float>(ball["y"].asNumber()),
            0.f  // z = ground level for 2D
        );
        gtBall.velocity = Vector3f(
            static_cast<float>(ball["vx"].asNumber()),
            static_cast<float>(ball["vy"].asNumber()),
            0.f
        );
        ws.balls.clear();
        ws.balls.push_back(gtBall);

        // Own team
        ws.ownTeamPlayers.clear();
        for (const auto& p : input["own_team"].asArray())
        {
            GroundTruthWorldState::GroundTruthPlayer player;
            player.number = p["number"].asInt();
            player.pose = Pose2f(
                static_cast<float>(p["theta"].asNumber()),
                static_cast<float>(p["x"].asNumber()),
                static_cast<float>(p["y"].asNumber())
            );
            player.upright = p["upright"].asBool(true);
            ws.ownTeamPlayers.push_back(player);
        }

        // Set own pose from the player this bridge controls.
        if (!ws.ownTeamPlayers.empty())
        {
            const auto it = std::find_if(
                ws.ownTeamPlayers.begin(), ws.ownTeamPlayers.end(),
                [this](const GroundTruthWorldState::GroundTruthPlayer& player)
                {
                    return player.number == playerNumber_;
                });
            ws.ownPose = it != ws.ownTeamPlayers.end() ? it->pose : ws.ownTeamPlayers.front().pose;
        }

        // Opponent team
        ws.opponentTeamPlayers.clear();
        for (const auto& p : input["opp_team"].asArray())
        {
            GroundTruthWorldState::GroundTruthPlayer player;
            player.number = p["number"].asInt();
            player.pose = Pose2f(
                static_cast<float>(p["theta"].asNumber()),
                static_cast<float>(p["x"].asNumber()),
                static_cast<float>(p["y"].asNumber())
            );
            player.upright = p["upright"].asBool(true);
            ws.opponentTeamPlayers.push_back(player);
        }
    }

    /**
     * Build GameControllerData from JSON input.
     */
    void buildGameControllerData(const JsonValue& input, unsigned currentBridgeTime, GameControllerData& gcd, const std::string& stateOverride = "")
    {
        const JsonArray& ownPlayers = input["own_team"].asArray();
        const JsonArray& oppPlayers = input["opp_team"].asArray();

        gcd.packetNumber = static_cast<uint8_t>(frameId_ & 0xff);
        gcd.playersPerTeam = static_cast<uint8_t>(numPlayers_);
        gcd.competitionPhase = COMPETITION_PHASE_ROUNDROBIN;
        gcd.competitionType = COMPETITION_TYPE_NORMAL;
        gcd.gamePhase = GAME_PHASE_NORMAL;
        gcd.setPlay = SET_PLAY_NONE;
        gcd.firstHalf = 1;
        gcd.kickingTeam = static_cast<uint8_t>(teamNumber_);
        gcd.secsRemaining = 600;
        gcd.secondaryTime = 0;
        gcd.timeLastPacketReceived = currentBridgeTime;
        gcd.isTrueData = true;

        std::string state = stateOverride.empty() ? input["game_state"].asString() : stateOverride;
        if (state == "playing")
            gcd.state = STATE_PLAYING;
        else if (state == "ready")
            gcd.state = STATE_READY;
        else if (state == "set")
            gcd.state = STATE_SET;
        else if (state == "initial")
            gcd.state = STATE_INITIAL;
        else if (state == "finished")
            gcd.state = STATE_FINISHED;
        else
            gcd.state = STATE_PLAYING;

        gcd.teams[0].teamNumber = static_cast<uint8_t>(teamNumber_);
        gcd.teams[0].fieldPlayerColor = static_cast<uint8_t>(Settings::blue);
        gcd.teams[0].goalkeeperColor = static_cast<uint8_t>(Settings::blue);
        gcd.teams[0].goalkeeper = 1;
        gcd.teams[0].score = static_cast<uint8_t>(input["own_score"].asInt());

        gcd.teams[1].teamNumber = static_cast<uint8_t>(teamNumber_ + 1);
        gcd.teams[1].fieldPlayerColor = static_cast<uint8_t>(Settings::red);
        gcd.teams[1].goalkeeperColor = static_cast<uint8_t>(Settings::red);
        gcd.teams[1].goalkeeper = 1;
        gcd.teams[1].score = static_cast<uint8_t>(input["opp_score"].asInt());

        for (int i = 0; i < MAX_NUM_PLAYERS; ++i)
        {
            gcd.teams[0].players[i].penalty = PENALTY_SUBSTITUTE;
            gcd.teams[0].players[i].secsTillUnpenalised = 0;
            gcd.teams[1].players[i].penalty = PENALTY_SUBSTITUTE;
            gcd.teams[1].players[i].secsTillUnpenalised = 0;
        }

        for(const auto& p : ownPlayers)
        {
            const int number = p["number"].asInt();
            if(number >= Settings::lowestValidPlayerNumber && number <= MAX_NUM_PLAYERS)
            {
                gcd.teams[0].players[number - Settings::lowestValidPlayerNumber].penalty = PENALTY_NONE;
                gcd.teams[0].players[number - Settings::lowestValidPlayerNumber].secsTillUnpenalised = 0;
            }
        }

        for(const auto& p : oppPlayers)
        {
            const int number = p["number"].asInt();
            if(number >= Settings::lowestValidPlayerNumber && number <= MAX_NUM_PLAYERS)
            {
                gcd.teams[1].players[number - Settings::lowestValidPlayerNumber].penalty = PENALTY_NONE;
                gcd.teams[1].players[number - Settings::lowestValidPlayerNumber].secsTillUnpenalised = 0;
            }
        }
    }

    /**
     * Read decisions from the Cognition thread.
     *
     * IMPLEMENTATION NOTE:
     * Direct Blackboard access across threads is not safe without
     * proper synchronization. The correct approach is to either:
     *
     * Option A: Use Debug module requests (mr command)
     *   - Send "mr Cognition:SkillRequest" via debug message
     *   - Read response from bridgeReceiver
     *   - More complex but thread-safe
     *
     * Option B: Use inter-thread ModulePacket
     *   - Configure additional representations to be sent from
     *     Cognition to our bridge via the sender/receiver system
     *   - Requires modifying threads.cfg or Configuration
     *
     * Option C: Direct Blackboard read with synchronization
     *   - Find the Cognition ModuleContainer
     *   - Call setGlobals() to switch to its Blackboard
     *   - Read representations
     *   - This is NOT thread-safe without locking
     *
     * For the initial implementation, we use Option C with a sleep-based
     * synchronization. This MUST be replaced with proper sync for production.
     */
    JsonValue readDecisions(int frameId)
    {
        JsonObject result;
        result["type"] = JsonValue("decision");
        result["frame_id"] = JsonValue(frameId);
        result["player_number"] = JsonValue(playerNumber_);
        result["debug_bridge_mode"] = JsonValue(bridgeMode_);

        // TODO: Implement proper cross-thread Blackboard reading.
        // For now, return placeholder values that indicate the bridge is active
        // but hasn't read real decisions yet.
        //
        // When proper reading is implemented, this section will:
        // 1. Access Cognition thread's Blackboard
        // 2. Read SkillRequest -> skill_request, target
        // 3. Read MotionRequest -> motion_request, speed
        // 4. Read ActivationGraph -> activation_graph nodes
        // 5. Read MotionInfo -> last_kick_type, last_kick_timestamp
        // 6. Read StrategyStatus -> tactic, role

        // Placeholder — indicates bridge is running but reading not yet wired
        result["skill_request"] = JsonValue("none");
        result["motion_request"] = JsonValue("stand");

        JsonArray motionSpeed;
        motionSpeed.push_back(JsonValue(0.0));
        motionSpeed.push_back(JsonValue(0.0));
        motionSpeed.push_back(JsonValue(0.0));
        result["motion_speed"] = JsonValue(motionSpeed);

        result["kick_type"] = JsonValue("none");
        result["kick_length_mm"] = JsonValue(0.0);
        result["last_kick_type"] = JsonValue("unavailable");
        result["last_kick_timestamp_ms"] = JsonValue(0);
        result["strategy_status"] = JsonValue("s4v4_scaled");
        result["current_tactic"] = JsonValue("none");
        result["behavior_status"] = JsonValue("unavailable");
        result["active_agent_number"] = JsonValue();

        result["activation_graph"] = JsonValue(JsonArray());
        result["debug_game_state"] = JsonValue("unavailable");
        result["debug_player_state"] = JsonValue("unavailable");
        result["debug_game_controller_active"] = JsonValue(false);
        result["debug_provider_game_state"] = JsonValue("unavailable");
        result["debug_provider_skill_request"] = JsonValue("unavailable");
        result["debug_provider_motion_request"] = JsonValue("unavailable");
        result["debug_provider_activation_graph"] = JsonValue("unavailable");
        result["debug_provider_behavior_status"] = JsonValue("unavailable");
        result["debug_provider_strategy_status"] = JsonValue("unavailable");
        result["debug_seen_messages_total"] = JsonValue(0);
        result["debug_recent_messages"] = JsonValue(JsonArray());

        if (!bridgeConsole_)
        {
            result["behavior_status"] = JsonValue("missing_bridge_console");
            return JsonValue(result);
        }

        verboseBridgeLog("[bridge] readDecisions: snapshot");
        const DecisionSnapshot snapshot = bridgeConsole_->latestSnapshot();
        const unsigned totalMessages = bridgeConsole_->totalReceivedMessages();
        const std::vector<std::string> recentMessages = bridgeConsole_->recentReceivedMessages();
        bool selfIsActive = false;

        result["debug_provider_game_state"] = JsonValue("debug_channel");
        result["debug_provider_skill_request"] = JsonValue("debug_channel");
        result["debug_provider_motion_request"] = JsonValue("debug_channel");
        result["debug_provider_motion_info"] = JsonValue("debug_channel");
        result["debug_provider_activation_graph"] = JsonValue("debug_channel");
        result["debug_provider_behavior_status"] = JsonValue("debug_channel");
        result["debug_provider_strategy_status"] = JsonValue("debug_channel");
        result["debug_completed_cognition_frames"] = JsonValue(static_cast<int>(snapshot.completedCognitionFrames));
        result["debug_completed_motion_frames"] = JsonValue(bridgeConsole_ ? static_cast<int>(bridgeConsole_->motionFrameFinishedCount()) : 0);
        result["debug_seen_messages_total"] = JsonValue(static_cast<int>(totalMessages));
        JsonArray recentMessageArray;
        for(const std::string& messageName : recentMessages)
            recentMessageArray.push_back(JsonValue(messageName));
        result["debug_recent_messages"] = JsonValue(recentMessageArray);

        if (snapshot.hasGameState)
        {
            const GameState& gs = snapshot.gameState;
            result["debug_game_state"] = JsonValue(enumNameOr(gs.state, "unavailable"));
            result["debug_player_state"] = JsonValue(enumNameOr(gs.playerState, "unavailable"));
            result["debug_game_controller_active"] = JsonValue(gs.gameControllerActive);
        }

        if (snapshot.hasSkillRequest)
        {
            verboseBridgeLog("[bridge] readDecisions: SkillRequest");
            const SkillRequest& sr = snapshot.skillRequest;
            result["skill_request"] = JsonValue(enumNameOr(sr.skill, "none"));

            if (sr.skill == SkillRequest::pass && sr.passTarget >= 0)
                result["pass_target_player_number"] = JsonValue(sr.passTarget);

            JsonObject skillTarget;
            skillTarget["x_mm"] = JsonValue(sr.target.translation.x());
            skillTarget["y_mm"] = JsonValue(sr.target.translation.y());
            skillTarget["theta_rad"] = JsonValue(sr.target.rotation);
            result["skill_target_pose"] = JsonValue(skillTarget);
            result["skill_target_direction_rad"] = JsonValue(sr.target.rotation);

            if (sr.skill == SkillRequest::shoot ||
                sr.skill == SkillRequest::pass ||
                sr.skill == SkillRequest::dribble ||
                sr.skill == SkillRequest::block)
            {
                selfIsActive = true;
            }
        }

        if (snapshot.hasMotionRequest)
        {
            verboseBridgeLog("[bridge] readDecisions: MotionRequest");
            const MotionRequest& mr = snapshot.motionRequest;
            result["motion_request"] = JsonValue(enumNameOr(mr.motion, "stand"));

            JsonArray walkSpeed;
            walkSpeed.push_back(JsonValue(mr.walkSpeed.translation.x()));
            walkSpeed.push_back(JsonValue(mr.walkSpeed.translation.y()));
            walkSpeed.push_back(JsonValue(mr.walkSpeed.rotation));
            result["motion_speed"] = JsonValue(walkSpeed);

            result["kick_type"] = JsonValue(enumNameOr(mr.kickType, "none"));
            result["kick_length_mm"] = JsonValue(mr.kickLength);

            JsonObject walkTarget;
            walkTarget["x_mm"] = JsonValue(mr.walkTarget.translation.x());
            walkTarget["y_mm"] = JsonValue(mr.walkTarget.translation.y());
            walkTarget["theta_rad"] = JsonValue(mr.walkTarget.rotation);
            result["motion_walk_target"] = JsonValue(walkTarget);
            result["motion_target_direction_rad"] = JsonValue(mr.targetDirection);
            result["motion_align_precisely"] = JsonValue(enumNameOr(mr.alignPrecisely, "notPrecise"));
            result["motion_pre_step_allowed"] = JsonValue(mr.preStepAllowed);
            result["motion_turn_kick_allowed"] = JsonValue(mr.turnKickAllowed);
        }

        if (snapshot.hasMotionInfo)
        {
            verboseBridgeLog("[bridge] readDecisions: MotionInfo");
            const MotionInfo& mi = snapshot.motionInfo;
            result["motion_executed_phase"] = JsonValue(enumNameOr(mi.executedPhase, "playDead"));

            JsonArray executedSpeed;
            executedSpeed.push_back(JsonValue(mi.speed.translation.x()));
            executedSpeed.push_back(JsonValue(mi.speed.translation.y()));
            executedSpeed.push_back(JsonValue(mi.speed.rotation));
            result["motion_executed_speed"] = JsonValue(executedSpeed);

            if (mi.lastKickType != KickInfo::numOfKickTypes)
                result["last_kick_type"] = JsonValue(enumNameOr(mi.lastKickType, "unavailable"));
            result["last_kick_timestamp_ms"] = JsonValue(static_cast<int>(mi.lastKickTimestamp));
        }

        if (snapshot.hasStrategyStatus)
        {
            verboseBridgeLog("[bridge] readDecisions: StrategyStatus");
            const StrategyStatus& ss = snapshot.strategyStatus;
            result["current_tactic"] = JsonValue(normalizeTacticName(enumNameOr(ss.acceptedTactic, "none")));
            result["strategy_proposed_tactic"] = JsonValue(normalizeTacticName(enumNameOr(ss.proposedTactic, "none")));
            result["strategy_role"] = JsonValue(enumNameOr(ss.role, "none"));
            result["strategy_position"] = JsonValue(enumNameOr(ss.position, "none"));
            result["strategy_proposed_set_play"] = JsonValue(enumNameOr(ss.proposedSetPlay, "none"));
            result["strategy_accepted_set_play"] = JsonValue(enumNameOr(ss.acceptedSetPlay, "none"));
            result["strategy_set_play_step"] = JsonValue(ss.setPlayStep);
            result["strategy_accepted_mirror"] = JsonValue(ss.acceptedMirror);
            result["strategy_proposed_mirror"] = JsonValue(ss.proposedMirror);

            if (Role::isActiveRole(ss.role))
                selfIsActive = true;
        }

        if (snapshot.hasBehaviorStatus)
        {
            verboseBridgeLog("[bridge] readDecisions: BehaviorStatus");
            const BehaviorStatus& bs = snapshot.behaviorStatus;

            if (bs.shootingTo.has_value())
                result["behavior_status"] = JsonValue("shooting");
            else if (bs.passTarget >= 0)
                result["behavior_status"] = JsonValue("passing");
            else
                result["behavior_status"] = JsonValue("playing");

            result["behavior_speed_mm_s"] = JsonValue(bs.speed);
            result["pass_target_player_number"] = JsonValue(bs.passTarget);
            result["pass_origin_player_number"] = JsonValue(bs.passOrigin);

            JsonObject walkingTo;
            walkingTo["x_mm"] = JsonValue(bs.walkingTo.x());
            walkingTo["y_mm"] = JsonValue(bs.walkingTo.y());
            result["behavior_walking_to"] = JsonValue(walkingTo);

            if (bs.shootingTo.has_value())
            {
                JsonObject shootingTo;
                shootingTo["x_mm"] = JsonValue(bs.shootingTo->x());
                shootingTo["y_mm"] = JsonValue(bs.shootingTo->y());
                result["behavior_shooting_to"] = JsonValue(shootingTo);
            }
        }

        if (snapshot.hasActivationGraph)
        {
            verboseBridgeLog("[bridge] readDecisions: ActivationGraph");
            JsonArray activationGraph;
            for (const ActivationGraph::Node& node : snapshot.activationGraph.graph)
            {
                JsonObject graphNode;
                graphNode["option"] = JsonValue(node.option);
                graphNode["depth"] = JsonValue(node.depth);
                graphNode["state"] = JsonValue(node.state);
                graphNode["option_time"] = JsonValue(node.optionTime);
                graphNode["state_time"] = JsonValue(node.stateTime);

                JsonArray parameters;
                for (const std::string& parameter : node.parameters)
                    parameters.push_back(JsonValue(parameter));
                graphNode["parameters"] = JsonValue(parameters);
                activationGraph.push_back(JsonValue(graphNode));
            }
            result["activation_graph"] = JsonValue(activationGraph);
        }

        const bool snapshotHadDecisionData =
            snapshot.hasGameState ||
            snapshot.hasSkillRequest ||
            snapshot.hasMotionRequest ||
            snapshot.hasMotionInfo ||
            snapshot.hasStrategyStatus ||
            snapshot.hasBehaviorStatus ||
            snapshot.hasActivationGraph;

        if(!snapshotHadDecisionData && cognition_)
        {
            verboseBridgeLog("[bridge] readDecisions: direct blackboard fallback");
            try
            {
                cognition_->setGlobals();
                Blackboard& bb = Blackboard::getInstance();

                result["debug_provider_game_state"] = JsonValue("cognition_blackboard");
                result["debug_provider_skill_request"] = JsonValue("cognition_blackboard");
                result["debug_provider_motion_request"] = JsonValue("cognition_blackboard");
                result["debug_provider_motion_info"] = JsonValue("cognition_blackboard");
                result["debug_provider_activation_graph"] = JsonValue("cognition_blackboard");
                result["debug_provider_behavior_status"] = JsonValue("cognition_blackboard");
                result["debug_provider_strategy_status"] = JsonValue("cognition_blackboard");

                if(bb.exists("GameState"))
                {
                    const GameState& gs = static_cast<const GameState&>(bb["GameState"]);
                    result["debug_game_state"] = JsonValue(enumNameOr(gs.state, "unavailable"));
                    result["debug_player_state"] = JsonValue(enumNameOr(gs.playerState, "unavailable"));
                    result["debug_game_controller_active"] = JsonValue(gs.gameControllerActive);
                }

                if(bb.exists("SkillRequest"))
                {
                    const SkillRequest& sr = static_cast<const SkillRequest&>(bb["SkillRequest"]);
                    result["skill_request"] = JsonValue(enumNameOr(sr.skill, "none"));
                    if (sr.skill == SkillRequest::pass && sr.passTarget >= 0)
                        result["pass_target_player_number"] = JsonValue(sr.passTarget);

                    JsonObject skillTarget;
                    skillTarget["x_mm"] = JsonValue(sr.target.translation.x());
                    skillTarget["y_mm"] = JsonValue(sr.target.translation.y());
                    skillTarget["theta_rad"] = JsonValue(sr.target.rotation);
                    result["skill_target_pose"] = JsonValue(skillTarget);
                    result["skill_target_direction_rad"] = JsonValue(sr.target.rotation);

                    if (sr.skill == SkillRequest::shoot ||
                        sr.skill == SkillRequest::pass ||
                        sr.skill == SkillRequest::dribble ||
                        sr.skill == SkillRequest::block)
                    {
                        selfIsActive = true;
                    }
                }

                if(bb.exists("MotionRequest"))
                {
                    const MotionRequest& mr = static_cast<const MotionRequest&>(bb["MotionRequest"]);
                    result["motion_request"] = JsonValue(enumNameOr(mr.motion, "stand"));

                    JsonArray walkSpeed;
                    walkSpeed.push_back(JsonValue(mr.walkSpeed.translation.x()));
                    walkSpeed.push_back(JsonValue(mr.walkSpeed.translation.y()));
                    walkSpeed.push_back(JsonValue(mr.walkSpeed.rotation));
                    result["motion_speed"] = JsonValue(walkSpeed);

                    result["kick_type"] = JsonValue(enumNameOr(mr.kickType, "none"));
                    result["kick_length_mm"] = JsonValue(mr.kickLength);

                    JsonObject walkTarget;
                    walkTarget["x_mm"] = JsonValue(mr.walkTarget.translation.x());
                    walkTarget["y_mm"] = JsonValue(mr.walkTarget.translation.y());
                    walkTarget["theta_rad"] = JsonValue(mr.walkTarget.rotation);
                    result["motion_walk_target"] = JsonValue(walkTarget);
                    result["motion_target_direction_rad"] = JsonValue(mr.targetDirection);
                    result["motion_align_precisely"] = JsonValue(enumNameOr(mr.alignPrecisely, "notPrecise"));
                    result["motion_pre_step_allowed"] = JsonValue(mr.preStepAllowed);
                    result["motion_turn_kick_allowed"] = JsonValue(mr.turnKickAllowed);
                }

                if(bb.exists("MotionInfo"))
                {
                    const MotionInfo& mi = static_cast<const MotionInfo&>(bb["MotionInfo"]);
                    result["motion_executed_phase"] = JsonValue(enumNameOr(mi.executedPhase, "playDead"));

                    JsonArray executedSpeed;
                    executedSpeed.push_back(JsonValue(mi.speed.translation.x()));
                    executedSpeed.push_back(JsonValue(mi.speed.translation.y()));
                    executedSpeed.push_back(JsonValue(mi.speed.rotation));
                    result["motion_executed_speed"] = JsonValue(executedSpeed);

                    if (mi.lastKickType != KickInfo::numOfKickTypes)
                        result["last_kick_type"] = JsonValue(enumNameOr(mi.lastKickType, "unavailable"));
                    result["last_kick_timestamp_ms"] = JsonValue(static_cast<int>(mi.lastKickTimestamp));
                }

                if(bb.exists("StrategyStatus"))
                {
                    const StrategyStatus& ss = static_cast<const StrategyStatus&>(bb["StrategyStatus"]);
                    result["current_tactic"] = JsonValue(normalizeTacticName(enumNameOr(ss.acceptedTactic, "none")));
                    result["strategy_proposed_tactic"] = JsonValue(normalizeTacticName(enumNameOr(ss.proposedTactic, "none")));
                    result["strategy_role"] = JsonValue(enumNameOr(ss.role, "none"));
                    result["strategy_position"] = JsonValue(enumNameOr(ss.position, "none"));
                    result["strategy_proposed_set_play"] = JsonValue(enumNameOr(ss.proposedSetPlay, "none"));
                    result["strategy_accepted_set_play"] = JsonValue(enumNameOr(ss.acceptedSetPlay, "none"));
                    result["strategy_set_play_step"] = JsonValue(ss.setPlayStep);
                    result["strategy_accepted_mirror"] = JsonValue(ss.acceptedMirror);
                    result["strategy_proposed_mirror"] = JsonValue(ss.proposedMirror);

                    if (Role::isActiveRole(ss.role))
                        selfIsActive = true;
                }

                if(bb.exists("BehaviorStatus"))
                {
                    const BehaviorStatus& bs = static_cast<const BehaviorStatus&>(bb["BehaviorStatus"]);
                    if (bs.shootingTo.has_value())
                        result["behavior_status"] = JsonValue("shooting");
                    else if (bs.passTarget >= 0)
                        result["behavior_status"] = JsonValue("passing");
                    else
                        result["behavior_status"] = JsonValue("playing");

                    result["behavior_speed_mm_s"] = JsonValue(bs.speed);
                    result["pass_target_player_number"] = JsonValue(bs.passTarget);
                    result["pass_origin_player_number"] = JsonValue(bs.passOrigin);

                    JsonObject walkingTo;
                    walkingTo["x_mm"] = JsonValue(bs.walkingTo.x());
                    walkingTo["y_mm"] = JsonValue(bs.walkingTo.y());
                    result["behavior_walking_to"] = JsonValue(walkingTo);

                    if (bs.shootingTo.has_value())
                    {
                        JsonObject shootingTo;
                        shootingTo["x_mm"] = JsonValue(bs.shootingTo->x());
                        shootingTo["y_mm"] = JsonValue(bs.shootingTo->y());
                        result["behavior_shooting_to"] = JsonValue(shootingTo);
                    }
                }

                if(bb.exists("ActivationGraph"))
                {
                    const ActivationGraph& ag = static_cast<const ActivationGraph&>(bb["ActivationGraph"]);
                    JsonArray activationGraph;
                    for (const ActivationGraph::Node& node : ag.graph)
                    {
                        JsonObject graphNode;
                        graphNode["option"] = JsonValue(node.option);
                        graphNode["depth"] = JsonValue(node.depth);
                        graphNode["state"] = JsonValue(node.state);
                        graphNode["option_time"] = JsonValue(node.optionTime);
                        graphNode["state_time"] = JsonValue(node.stateTime);

                        JsonArray parameters;
                        for (const std::string& parameter : node.parameters)
                            parameters.push_back(JsonValue(parameter));
                        graphNode["parameters"] = JsonValue(parameters);
                        activationGraph.push_back(JsonValue(graphNode));
                    }
                    result["activation_graph"] = JsonValue(activationGraph);
                }
            }
            catch(const std::exception& e)
            {
                result["debug_blackboard_error"] = JsonValue(e.what());
            }
            catch(...)
            {
                result["debug_blackboard_error"] = JsonValue("unknown");
            }

            if(bridgeConsole_)
                bridgeConsole_->setGlobals();
        }

        if(motion_)
        {
            try
            {
                motion_->setGlobals();
                Blackboard& motionBb = Blackboard::getInstance();
                if(motionBb.exists("MotionInfo"))
                {
                    const MotionInfo& mi = static_cast<const MotionInfo&>(motionBb["MotionInfo"]);
                    result["debug_provider_motion_info"] = JsonValue("motion_blackboard");
                    result["motion_executed_phase"] = JsonValue(enumNameOr(mi.executedPhase, "playDead"));

                    JsonArray executedSpeed;
                    executedSpeed.push_back(JsonValue(mi.speed.translation.x()));
                    executedSpeed.push_back(JsonValue(mi.speed.translation.y()));
                    executedSpeed.push_back(JsonValue(mi.speed.rotation));
                    result["motion_executed_speed"] = JsonValue(executedSpeed);

                    if(mi.lastKickType != KickInfo::numOfKickTypes)
                        result["last_kick_type"] = JsonValue(enumNameOr(mi.lastKickType, "unavailable"));
                    result["last_kick_timestamp_ms"] = JsonValue(static_cast<int>(mi.lastKickTimestamp));
                }
            }
            catch(const std::exception& e)
            {
                result["debug_motion_blackboard_error"] = JsonValue(e.what());
            }
            catch(...)
            {
                result["debug_motion_blackboard_error"] = JsonValue("unknown");
            }

            if(cognition_)
                cognition_->setGlobals();
            else if(bridgeConsole_)
                bridgeConsole_->setGlobals();
        }

        if (selfIsActive)
            result["active_agent_number"] = JsonValue(playerNumber_);

        verboseBridgeLog("[bridge] readDecisions: done");

        return JsonValue(result);
    }

    /**
     * PRODUCTION IMPLEMENTATION for readDecisions (uncomment when ready):
     *
     * This implementation directly accesses the Cognition thread's Blackboard.
     * Requires:
     *   1. Finding the Cognition ModuleContainer in the Robot's thread list
     *   2. Proper synchronization (the sleep in processFrame must be sufficient
     *      for cognition to complete, or use an event/semaphore)
     *
     * JsonValue readDecisionsFromBlackboard(int frameId)
     * {
     *     // Find Cognition thread
     *     ModuleContainer* cognition = nullptr;
     *     for (auto* thread = robot_->front(); thread; thread = thread->getNext())
     *     {
     *         ModuleContainer* mc = dynamic_cast<ModuleContainer*>(thread);
     *         if (mc && mc->getName() == "Cognition")
     *         {
     *             cognition = mc;
     *             break;
     *         }
     *     }
     *     if (!cognition)
     *         return makeError("Cognition thread not found");
     *
     *     // Switch to Cognition's Blackboard
     *     cognition->setGlobals();
     *     Blackboard& bb = Blackboard::getInstance();
     *
     *     JsonObject result;
     *     result["type"] = JsonValue("decision");
     *     result["frame_id"] = JsonValue(frameId);
     *
     *     // Read SkillRequest
     *     if (bb.exists("SkillRequest"))
     *     {
     *         const SkillRequest& sr = static_cast<const SkillRequest&>(bb["SkillRequest"]);
     *         result["skill_request"] = JsonValue(TypeRegistry::getEnumName(sr.skill));
     *         result["player_number"] = JsonValue(sr.passTarget);
     *     }
     *
     *     // Read MotionRequest
     *     if (bb.exists("MotionRequest"))
     *     {
     *         const MotionRequest& mr = static_cast<const MotionRequest&>(bb["MotionRequest"]);
     *         result["motion_request"] = JsonValue(TypeRegistry::getEnumName(mr.motion));
     *     }
     *
     *     // Read ActivationGraph
     *     if (bb.exists("ActivationGraph"))
     *     {
     *         const ActivationGraph& ag = static_cast<const ActivationGraph&>(bb["ActivationGraph"]);
     *         JsonArray graph;
     *         for (const auto& node : ag.graph)
     *         {
     *             JsonObject n;
     *             n["option"] = JsonValue(node.option);
     *             n["depth"] = JsonValue(node.depth);
     *             n["state"] = JsonValue(node.state);
     *             n["option_time"] = JsonValue(node.optionTime);
     *             n["state_time"] = JsonValue(node.stateTime);
     *             JsonArray params;
     *             for (const auto& p : node.parameters)
     *                 params.push_back(JsonValue(p));
     *             n["parameters"] = JsonValue(params);
     *             graph.push_back(JsonValue(n));
     *         }
     *         result["activation_graph"] = JsonValue(graph);
     *     }
     *
     *     // Read StrategyStatus
     *     if (bb.exists("StrategyStatus"))
     *     {
     *         const StrategyStatus& ss = static_cast<const StrategyStatus&>(bb["StrategyStatus"]);
     *         result["current_tactic"] = JsonValue(TypeRegistry::getEnumName(ss.acceptedTactic));
     *         result["strategy_status"] = JsonValue(TypeRegistry::getEnumName(ss.role));
     *     }
     *
     *     // Read MotionInfo (for last_kick_type)
     *     if (bb.exists("MotionInfo"))
     *     {
     *         const MotionInfo& mi = static_cast<const MotionInfo&>(bb["MotionInfo"]);
     *         result["last_kick_type"] = JsonValue(TypeRegistry::getEnumName(mi.lastKickType));
     *         result["last_kick_timestamp_ms"] = JsonValue(static_cast<int>(mi.lastKickTimestamp));
     *     }
     *
     *     return JsonValue(result);
     * }
     */

    JsonValue makeError(const std::string& message)
    {
        JsonObject obj;
        obj["type"] = JsonValue("error");
        obj["message"] = JsonValue(message);
        return JsonValue(obj);
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// MAIN — stdin/stdout JSON-line protocol loop
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
#ifdef WINDOWS
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(rlBridgeUnhandledExceptionFilter);
#endif
    std::cerr << "[bridge] BHuman Bridge v1.0 starting" << std::endl;

    BridgeRunner bridge;
    std::string line;

    while (std::getline(std::cin, line))
    {
        if (line.empty()) continue;

        auto parsed = JsonParser::parse(line);
        if (!parsed)
        {
            JsonObject err;
            err["type"] = JsonValue("error");
            err["message"] = JsonValue("Invalid JSON input");
            std::cout << jsonSerialize(JsonValue(err)) << std::endl;
            std::cout.flush();
            continue;
        }

        const JsonValue& msg = *parsed;
        std::string msgType = msg["type"].asString();

        if (msgType == "init")
        {
            std::string scenario = msg["scenario"].asString();
            std::string location = msg["location"].asString();
            std::string bridgeMode = msg["bridge_mode"].asString();
            int teamNumber = msg["team_number"].asInt(1);
            int playerNumber = msg["player_number"].asInt(1);
            int numPlayers = msg["num_players"].asInt(4);

            if (scenario.empty()) scenario = "2D";
            if (location.empty()) location = "4v4_Scaled";
            if (bridgeMode.empty()) bridgeMode = "persistent_2d_cognition";

            bool ok = bridge.init(scenario, location, teamNumber, playerNumber, numPlayers, bridgeMode);

            JsonObject response;
            if (ok)
            {
                response["type"] = JsonValue("ready");
                response["version"] = JsonValue("1.0");
                response["bridge_mode"] = JsonValue(bridgeMode);
                response["scenario"] = JsonValue(scenario);
                response["location"] = JsonValue(location);
            }
            else
            {
                response["type"] = JsonValue("error");
                response["message"] = JsonValue("Failed to initialize BHuman Robot");
            }
            std::cout << jsonSerialize(JsonValue(response)) << std::endl;
            std::cout.flush();
        }
        else if (msgType == "frame")
        {
            JsonValue result = bridge.processFrame(msg);
            std::cout << jsonSerialize(result) << std::endl;
            std::cout.flush();
        }
        else if (msgType == "reset")
        {
            bridge.reset();
            JsonObject ack;
            ack["type"] = JsonValue("ack");
            ack["command"] = JsonValue("reset");
            std::cout << jsonSerialize(JsonValue(ack)) << std::endl;
            std::cout.flush();
        }
        else if (msgType == "shutdown")
        {
            bridge.shutdown();
            break;
        }
        else
        {
            JsonObject err;
            err["type"] = JsonValue("error");
            err["message"] = JsonValue("Unknown message type: " + msgType);
            std::cout << jsonSerialize(JsonValue(err)) << std::endl;
            std::cout.flush();
        }
    }

    std::cerr << "[bridge] Exiting" << std::endl;
    return 0;
}

