
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"

#include "whisper.h"
#include "grammar-parser.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <cfloat>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// helper function to replace substrings
static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    for (size_t pos = 0; ; pos += replace.length()) {
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

// command-line parameters
struct whisper_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors  = 1;
    int32_t offset_t_ms   = 0;
    int32_t offset_n      = 0;
    int32_t duration_ms   = 0;
    int32_t progress_step = 5;
    int32_t max_context   = -1;
    int32_t max_len       = 0;
    int32_t best_of       = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size     = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
    
    int32_t command_ms    = 3000;
    int32_t buffer_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t audio_ctx     = 0;

    float vad_thold  = 0.4f;
    float freq_thold = 100.0f;

    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.6f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_confidence= false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool suppress_nst    = false;

    bool print_energy    = false;

    std::string language  = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model     = "models/ggml-base.en.bin";
    std::string grammar;
    std::string grammar_rule;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;

    std::string openvino_encode_device = "CPU";

    grammar_parser::parse_state grammar_parsed;

    // Voice Activity Detection (VAD) parameters
    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
};

static void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static char * whisper_param_turn_lowercase(char * in){
    int string_len = strlen(in);
    for (int i = 0; i < string_len; i++){
        *(in+i) = tolower((unsigned char)*(in+i));
    }
    return in;
}

static char * requires_value_error(const std::string & arg) {
    fprintf(stderr, "error: argument %s requires value\n", arg.c_str());
    exit(0);
}

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        #define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))
        else if (arg == "-t"    || arg == "--threads")         { params.n_threads       = std::stoi(ARGV_NEXT); }
        else if (arg == "-p"    || arg == "--processors")      { params.n_processors    = std::stoi(ARGV_NEXT); }
        else if (arg == "-ot"   || arg == "--offset-t")        { params.offset_t_ms     = std::stoi(ARGV_NEXT); }
        else if (arg == "-on"   || arg == "--offset-n")        { params.offset_n        = std::stoi(ARGV_NEXT); }
        else if (arg == "-d"    || arg == "--duration")        { params.duration_ms     = std::stoi(ARGV_NEXT); }
        else if (arg == "-mc"   || arg == "--max-context")     { params.max_context     = std::stoi(ARGV_NEXT); }
        else if (arg == "-ml"   || arg == "--max-len")         { params.max_len         = std::stoi(ARGV_NEXT); }
        else if (arg == "-bo"   || arg == "--best-of")         { params.best_of         = std::stoi(ARGV_NEXT); }
        else if (arg == "-bs"   || arg == "--beam-size")       { params.beam_size       = std::stoi(ARGV_NEXT); }
        else if (arg == "-ac"   || arg == "--audio-ctx")       { params.audio_ctx       = std::stoi(ARGV_NEXT); }
        else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(ARGV_NEXT); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(ARGV_NEXT); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(ARGV_NEXT); }
        else if (arg == "-nth"  || arg == "--no-speech-thold") { params.no_speech_thold = std::stof(ARGV_NEXT); }
        else if (arg == "-tp"   || arg == "--temperature")     { params.temperature     = std::stof(ARGV_NEXT); }
        else if (arg == "-tpi"  || arg == "--temperature-inc") { params.temperature_inc = std::stof(ARGV_NEXT); }
        else if (arg == "-debug"|| arg == "--debug-mode")      { params.debug_mode      = true; }
        else if (arg == "-tr"   || arg == "--translate")       { params.translate       = true; }
        else if (arg == "-di"   || arg == "--diarize")         { params.diarize         = true; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")     { params.tinydiarize     = true; }
        else if (arg == "-sow"  || arg == "--split-on-word")   { params.split_on_word   = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")     { params.no_fallback     = true; }
        else if (arg == "-fp"   || arg == "--font-path")       { params.font_path       = ARGV_NEXT; }
        else if (arg == "-np"   || arg == "--no-prints")       { params.no_prints       = true; }
        else if (arg == "-ps"   || arg == "--print-special")   { params.print_special   = true; }
        else if (arg == "-pc"   || arg == "--print-colors")    { params.print_colors    = true; }
        else if (                  arg == "--print-confidence"){ params.print_confidence= true; }
        else if (arg == "-pp"   || arg == "--print-progress")  { params.print_progress  = true; }
        else if (arg == "-nt"   || arg == "--no-timestamps")   { params.no_timestamps   = true; }
        else if (arg == "-l"    || arg == "--language")        { params.language        = whisper_param_turn_lowercase(ARGV_NEXT); }
        else if (arg == "-dl"   || arg == "--detect-language") { params.detect_language = true; }
        else if (                  arg == "--prompt")          { params.prompt          = ARGV_NEXT; }
        else if (arg == "-m"    || arg == "--model")           { params.model           = ARGV_NEXT; }
        else if (arg == "-oved" || arg == "--ov-e-device")     { params.openvino_encode_device = ARGV_NEXT; }
        else if (arg == "-ls"   || arg == "--log-score")       { params.log_score       = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")          { params.use_gpu         = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")      { params.flash_attn      = true; }
        else if (arg == "-sns"  || arg == "--suppress-nst")    { params.suppress_nst    = true; }
        else if (                  arg == "--suppress-regex")  { params.suppress_regex  = ARGV_NEXT; }
        else if (                  arg == "--grammar")         { params.grammar         = ARGV_NEXT; }
        else if (                  arg == "--grammar-rule")    { params.grammar_rule    = ARGV_NEXT; }
        else if (                  arg == "--grammar-penalty") { params.grammar_penalty = std::stof(ARGV_NEXT); }
        // Voice Activity Detection (VAD)
        else if (                  arg == "--vad")                         { params.vad                         = true; }
        else if (arg == "-vm"   || arg == "--vad-model")                   { params.vad_model                   = ARGV_NEXT; }
        else if (arg == "-vt"   || arg == "--vad-threshold")               { params.vad_threshold               = std::stof(ARGV_NEXT); }
        else if (arg == "-vsd"  || arg == "--vad-min-speech-duration-ms")  { params.vad_min_speech_duration_ms  = std::stoi(ARGV_NEXT); }
        else if (arg == "-vsd"  || arg == "--vad-min-silence-duration-ms") { params.vad_min_speech_duration_ms  = std::stoi(ARGV_NEXT); }
        else if (arg == "-vmsd" || arg == "--vad-max-speech-duration-s")   { params.vad_max_speech_duration_s   = std::stof(ARGV_NEXT); }
        else if (arg == "-vp"   || arg == "--vad-speech-pad-ms")           { params.vad_speech_pad_ms           = std::stoi(ARGV_NEXT); }
        else if (arg == "-vo"   || arg == "--vad-samples-overlap")         { params.vad_samples_overlap         = std::stof(ARGV_NEXT); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options] file0 file1 ...\n", argv[0]);
    fprintf(stderr, "supported audio formats: flac, mp3, ogg, wav\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,        --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,      --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "  -p N,      --processors N      [%-7d] number of processors to use during computation\n", params.n_processors);
    fprintf(stderr, "  -ot N,     --offset-t N        [%-7d] time offset in milliseconds\n",                    params.offset_t_ms);
    fprintf(stderr, "  -on N,     --offset-n N        [%-7d] segment index offset\n",                           params.offset_n);
    fprintf(stderr, "  -d  N,     --duration N        [%-7d] duration of audio to process in milliseconds\n",   params.duration_ms);
    fprintf(stderr, "  -mc N,     --max-context N     [%-7d] maximum number of text context tokens to store\n", params.max_context);
    fprintf(stderr, "  -ml N,     --max-len N         [%-7d] maximum segment length in characters\n",           params.max_len);
    fprintf(stderr, "  -sow,      --split-on-word     [%-7s] split on word rather than on token\n",             params.split_on_word ? "true" : "false");
    fprintf(stderr, "  -bo N,     --best-of N         [%-7d] number of best candidates to keep\n",              params.best_of);
    fprintf(stderr, "  -bs N,     --beam-size N       [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -ac N,     --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -nth N,    --no-speech-thold N [%-7.2f] no speech threshold\n",                          params.no_speech_thold);
    fprintf(stderr, "  -tp,       --temperature N     [%-7.2f] The sampling temperature, between 0 and 1\n",    params.temperature);
    fprintf(stderr, "  -tpi,      --temperature-inc N [%-7.2f] The increment of temperature, between 0 and 1\n",params.temperature_inc);
    fprintf(stderr, "  -debug,    --debug-mode        [%-7s] enable debug mode (eg. dump log_mel)\n",           params.debug_mode ? "true" : "false");
    fprintf(stderr, "  -tr,       --translate         [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -di,       --diarize           [%-7s] stereo audio diarization\n",                       params.diarize ? "true" : "false");
    fprintf(stderr, "  -tdrz,     --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -nf,       --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -fp,       --font-path         [%-7s] path to a monospace font for karaoke video\n",     params.font_path.c_str());
    fprintf(stderr, "  -of FNAME, --output-file FNAME [%-7s] output file path (without file extension)\n",      "");
    fprintf(stderr, "  -np,       --no-prints         [%-7s] do not print anything other than the results\n",   params.no_prints ? "true" : "false");
    fprintf(stderr, "  -ps,       --print-special     [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -pc,       --print-colors      [%-7s] print colors\n",                                   params.print_colors ? "true" : "false");
    fprintf(stderr, "             --print-confidence  [%-7s] print confidence\n",                               params.print_confidence ? "true" : "false");
    fprintf(stderr, "  -pp,       --print-progress    [%-7s] print progress\n",                                 params.print_progress ? "true" : "false");
    fprintf(stderr, "  -nt,       --no-timestamps     [%-7s] do not print timestamps\n",                        params.no_timestamps ? "true" : "false");
    fprintf(stderr, "  -l LANG,   --language LANG     [%-7s] spoken language ('auto' for auto-detect)\n",       params.language.c_str());
    fprintf(stderr, "  -dl,       --detect-language   [%-7s] exit after automatically detecting language\n",    params.detect_language ? "true" : "false");
    fprintf(stderr, "             --prompt PROMPT     [%-7s] initial prompt (max n_text_ctx/2 tokens)\n",       params.prompt.c_str());
    fprintf(stderr, "  -m FNAME,  --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input audio file path\n",                            "");
    fprintf(stderr, "  -oved D,   --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
    fprintf(stderr, "  -ls,       --log-score         [%-7s] log best decoder scores of tokens\n",              params.log_score?"true":"false");
    fprintf(stderr, "  -ng,       --no-gpu            [%-7s] disable GPU\n",                                    params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,       --flash-attn        [%-7s] flash attention\n",                                params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -sns,      --suppress-nst      [%-7s] suppress non-speech tokens\n",                     params.suppress_nst ? "true" : "false");
    fprintf(stderr, "  --suppress-regex REGEX         [%-7s] regular expression matching tokens to suppress\n", params.suppress_regex.c_str());
    fprintf(stderr, "  --grammar GRAMMAR              [%-7s] GBNF grammar to guide decoding\n",                 params.grammar.c_str());
    fprintf(stderr, "  --grammar-rule RULE            [%-7s] top-level GBNF grammar rule name\n",               params.grammar_rule.c_str());
    fprintf(stderr, "  --grammar-penalty N            [%-7.1f] scales down logits of nongrammar tokens\n",      params.grammar_penalty);
    // Voice Activity Detection (VAD) parameters
    fprintf(stderr, "\nVoice Activity Detection (VAD) options:\n");
    fprintf(stderr, "             --vad                           [%-7s] enable Voice Activity Detection (VAD)\n",            params.vad ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME               [%-7s] VAD model path\n",                                   params.vad_model.c_str());
    fprintf(stderr, "  -vt N,     --vad-threshold N               [%-7.2f] VAD threshold for speech recognition\n",           params.vad_threshold);
    fprintf(stderr, "  -vspd N,   --vad-min-speech-duration-ms  N [%-7d] VAD min speech duration (0.0-1.0)\n",                params.vad_min_speech_duration_ms);
    fprintf(stderr, "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",      params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",      params.vad_max_speech_duration_s == FLT_MAX ?
                                                                                                                                  std::string("FLT_MAX").c_str() :
                                                                                                                                  std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",             params.vad_speech_pad_ms);
    fprintf(stderr, "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n", params.vad_samples_overlap);
    fprintf(stderr, "\n");
}

struct whisper_print_user_data {
    const whisper_params * params;

    int progress_prev;
};

static void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev  = &(((whisper_print_user_data *) user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

int main(int argc, char ** argv) {
#if defined(_WIN32)
    // Set the console output code page to UTF-8, while command line arguments
    // are still encoded in the system's code page. In this way, we can print
    // non-ASCII characters to the console, and access files with non-ASCII paths.
    SetConsoleOutputCP(CP_UTF8);
#endif

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        whisper_print_usage(argc, argv, params);
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    if (params.diarize && params.tinydiarize) {
        fprintf(stderr, "error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);

    if (!params.grammar.empty()) {
        auto & grammar = params.grammar_parsed;
        if (is_file_exist(params.grammar.c_str())) {
            // read grammar from file
            std::ifstream ifs(params.grammar.c_str());
            const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            grammar = grammar_parser::parse(txt.c_str());
        } else {
            // read grammar from string
            grammar = grammar_parser::parse(params.grammar.c_str());
        }

        // will be empty (default) if there are parse errors
        if (grammar.rules.empty()) {
            fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
            return 4;
        } else {
            fprintf(stderr, "%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            fprintf(stderr, "\n");
        }
    }

    if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
            params.language = "en";
            params.translate = false;
            fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
        }
    }

    if (params.detect_language) {
        params.language = "auto";
    }

    if (!params.no_prints) {
        // print system information
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());

        // print some info about the processing
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing -> %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                __func__, params.n_threads, params.n_processors, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.tinydiarize ? "tdrz = 1, " : "",
                params.no_timestamps ? 0 : 1);

        if (params.print_colors) {
            fprintf(stderr, "%s: color scheme: red (low confidence), yellow (medium), green (high confidence)\n", __func__);
        } else if (params.print_confidence) {
            fprintf(stderr, "%s: confidence: highlighted (low confidence), underlined (medium), dim (high confidence)\n", __func__);
        }
        fprintf(stderr, "\n");
    }

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
        wparams.strategy = (params.beam_size > 1 || use_grammar) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = true;
        wparams.print_progress   = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.print_special    = params.print_special;
        wparams.translate        = params.translate;
        wparams.language         = params.language.c_str();
        wparams.detect_language  = params.detect_language;
        wparams.n_threads        = params.n_threads;
        wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.token_timestamps = params.max_len > 0;
        wparams.thold_pt         = params.word_thold;
        wparams.max_len          = params.max_len == 0 ? 60 : params.max_len;
        wparams.split_on_word    = params.split_on_word;
        wparams.audio_ctx        = params.audio_ctx;

        wparams.debug_mode       = params.debug_mode;

        wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

        wparams.suppress_regex   = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();

        wparams.initial_prompt   = params.prompt.c_str();

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.temperature_inc  = params.no_fallback ? 0.0f : params.temperature_inc;
        wparams.temperature      = params.temperature;

        wparams.entropy_thold    = params.entropy_thold;
        wparams.logprob_thold    = params.logprob_thold;
        wparams.no_speech_thold  = params.no_speech_thold;

        wparams.no_timestamps    = params.no_timestamps;

        wparams.suppress_nst     = params.suppress_nst;

        wparams.vad            = params.vad;
        wparams.vad_model_path = params.vad_model.c_str();

        wparams.vad_params.threshold               = params.vad_threshold;
        wparams.vad_params.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
        wparams.vad_params.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        wparams.vad_params.max_speech_duration_s   = params.vad_max_speech_duration_s;
        wparams.vad_params.speech_pad_ms           = params.vad_speech_pad_ms;
        wparams.vad_params.samples_overlap         = params.vad_samples_overlap;

        whisper_print_user_data user_data = { &params, 0 };

        const auto & grammar_parsed = params.grammar_parsed;
        auto grammar_rules = grammar_parsed.c_rules();

        if (use_grammar) {
            if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end()) {
                fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
            } else {
                wparams.grammar_rules = grammar_rules.data();
                wparams.n_grammar_rules = grammar_rules.size();
                wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
                wparams.grammar_penalty = params.grammar_penalty;
            }
        }

        if (wparams.print_progress) {
            wparams.progress_callback           = whisper_print_progress_callback;
            wparams.progress_callback_user_data = &user_data;
        }

        // examples for abort mechanism
        // in examples below, we do not abort the processing, but we could if the flag is set to true

        // the callback is called before every encoder run - if it returns false, the processing is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
                bool is_aborted = *(bool*)user_data;
                return !is_aborted;
            };
            wparams.encoder_begin_callback_user_data = &is_aborted;
        }

        // the callback is called before every computation - if it returns true, the computation is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.abort_callback = [](void * user_data) {
                bool is_aborted = *(bool*)user_data;
                return is_aborted;
            };
            wparams.abort_callback_user_data = &is_aborted;
        }

        std::vector<float> pcmf32;               // mono-channel F32 PCM

        // init audio
        audio_async audio(params.buffer_ms);
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }
        
        audio.resume();

        // wait for 1 second to avoid any buffered noise
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        audio.clear();

        bool is_running = true;
        float logprob_min0 = 0.0f;
        float logprob_min  = 0.0f;

        float logprob_sum0 = 0.0f;
        float logprob_sum  = 0.0f;

        int n_tokens0 = 0;
        int n_tokens  = 0;

        fprintf(stdout, "%s: Athenea ...\n", __func__);

        while (is_running)
        {
            // handle Ctrl + C
            is_running = sdl_poll_events();

            audio.get(1500, pcmf32);

            if (::vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy)) {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                int64_t t_ms = 0;

                // we have heard the activation phrase, now detect the commands
                audio.get(params.command_ms, pcmf32);

                fprintf(stderr, "%s: voice -> %d samples, %.1f sec ...\n", __func__, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE);

                const auto t_start = std::chrono::high_resolution_clock::now();
                std::string result = "";
                if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) == 0) {
                    const int n_segments = whisper_full_n_segments(ctx);

                    for (int i = 0; i < n_segments; ++i) {
                        const char * text = whisper_full_get_segment_text(ctx, i);
    
                        result += text;
    
                        const int n = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < n; ++j) {
                            const auto token = whisper_full_get_token_data(ctx, i, j);
    
                            if(token.plog > 0.0f)
                                exit(0);
                            logprob_min = std::min(logprob_min, token.plog);
                            logprob_sum += token.plog;
                            ++n_tokens;
                        }
                    }
    
                    const auto t_end = std::chrono::high_resolution_clock::now();
                    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

                    const float p = 100.0f * std::exp(logprob_min);
                    fprintf(stdout, "%s:   DEBUG: txt = '%s', prob = %.2f%%, (t = %lld ms)\n", __func__, result.c_str(), p, (long long)t_ms);
                }

                audio.clear();
            }
        }
    }

    if (!params.no_prints) {
        whisper_print_timings(ctx);
    }
    whisper_free(ctx);

    return 0;
}
