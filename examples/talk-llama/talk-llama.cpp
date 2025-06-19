// Talk with AI
//

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "grammar-parser.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_bos, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

static std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
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
    int32_t buffer_ms     = 30000;
    int32_t keep_ms       = 200;
    int32_t capture_id    = -1;
    int32_t audio_ctx     = 0;
    int32_t n_gpu_layers  = 999;

    float vad_thold  = 0.4f;
    float freq_thold = 100.0f;
    float word_thold      =  0.01f;
    float entropy_thold   =  2.40f;
    float logprob_thold   = -1.00f;
    float no_speech_thold =  0.6f;
    float grammar_penalty = 100.0f;
    float temperature     = 0.0f;
    float temperature_inc = 0.2f;

    bool print_energy    = false;
    bool verbose_prompt  = false;
    bool print_special   = false;
    bool no_timestamps   = true;
    bool use_gpu         = true;
    bool flash_attn      = false;
    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool no_prints       = false;
    bool print_colors    = false;
    bool print_confidence= false;
    bool print_progress  = false;
    bool log_score       = false;
    bool suppress_nst    = false;

    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
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
    
    std::string person       = "Georgi";
    std::string bot_name     = "Aura";
    std::string language     = "en";
    std::string model_wsp    = "/etc/models/ggml-tiny.en.bin";
    std::string model_llama  = "/etc/models/gemma-3-1b-it-Q4_K_M.gguf";
    std::string speak        = "/etc/talk-llama/speak";
    std::string speak_file   = "/etc/talk-llama/to_speak.txt";
    std::string prompt       = "";
    std::string path_session = "";       // path to file for saving/loading model eval state
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

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
        else if (arg == "-c"   || arg == "--capture")          { params.capture_id      = std::stoi(ARGV_NEXT); }
        else if (arg == "-ngl" || arg == "--n-gpu-layers")     { params.n_gpu_layers    = std::stoi(ARGV_NEXT); }
        else if (arg == "-vth" || arg == "--vad-thold")        { params.vad_thold       = std::stof(ARGV_NEXT); }
        else if (arg == "-fth" || arg == "--freq-thold")       { params.freq_thold      = std::stof(ARGV_NEXT); }
        else if (arg == "-wt"   || arg == "--word-thold")      { params.word_thold      = std::stof(ARGV_NEXT); }
        else if (arg == "-et"   || arg == "--entropy-thold")   { params.entropy_thold   = std::stof(ARGV_NEXT); }
        else if (arg == "-lpt"  || arg == "--logprob-thold")   { params.logprob_thold   = std::stof(ARGV_NEXT); }
        else if (arg == "-nth"  || arg == "--no-speech-thold") { params.no_speech_thold = std::stof(ARGV_NEXT); }
        else if (arg == "-tp"   || arg == "--temperature")     { params.temperature     = std::stof(ARGV_NEXT); }
        else if (arg == "-tpi"  || arg == "--temperature-inc") { params.temperature_inc = std::stof(ARGV_NEXT); }
        else if (arg == "-pe"  || arg == "--print-energy")     { params.print_energy    = true; }
        else if (arg == "-vp"  || arg == "--verbose-prompt")   { params.verbose_prompt  = true; }
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
        else if (arg == "-pn"   || arg == "--person")           { params.person          = argv[++i]; }
        else if (arg == "-bn"   || arg == "--bot-name")         { params.bot_name        = argv[++i]; }
        else if (arg == "-mw"   || arg == "--model-whisper")    { params.model_wsp       = argv[++i]; }
        else if (arg == "-mll"  || arg == "--model-llama")      { params.model_llama     = argv[++i]; }
        else if (arg == "-s"    || arg == "--speak")            { params.speak           = argv[++i]; }
        else if (arg == "-sf"   || arg == "--speak-file")       { params.speak_file      = argv[++i]; }
        else if (arg == "--prompt-file")                        {
            std::ifstream file(argv[++i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
        else if (arg == "--session")                           { params.path_session    = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N          [%-7d] number of threads to use during computation\n",    params.n_threads);
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
    fprintf(stderr, "  -c ID,    --capture ID         [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N     [%-7d] number of layers to store in VRAM\n",              params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N        [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N       [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -wt N,     --word-thold N      [%-7.2f] word timestamp probability threshold\n",         params.word_thold);
    fprintf(stderr, "  -et N,     --entropy-thold N   [%-7.2f] entropy threshold for decoder fail\n",           params.entropy_thold);
    fprintf(stderr, "  -lpt N,    --logprob-thold N   [%-7.2f] log probability threshold for decoder fail\n",   params.logprob_thold);
    fprintf(stderr, "  -nth N,    --no-speech-thold N [%-7.2f] no speech threshold\n",                          params.no_speech_thold);
    fprintf(stderr, "  -tp,       --temperature N     [%-7.2f] The sampling temperature, between 0 and 1\n",    params.temperature);
    fprintf(stderr, "  -tpi,      --temperature-inc N [%-7.2f] The increment of temperature, between 0 and 1\n",params.temperature_inc);
    fprintf(stderr, "  -pe,      --print-energy       [%-7s] print sound energy (for debugging)\n",             params.print_energy ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt     [%-7s] print prompt at start\n",                          params.verbose_prompt ? "true" : "false");
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
    fprintf(stderr, "  -f FNAME,  --file FNAME        [%-7s] input audio file path\n",                          "");
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
    fprintf(stderr, "  -vsd N,    --vad-min-silence-duration-ms N [%-7d] VAD min silence duration (to split segments)\n",     params.vad_min_silence_duration_ms);
    fprintf(stderr, "  -vmsd N,   --vad-max-speech-duration-s   N [%-7s] VAD max speech duration (auto-split longer)\n",      params.vad_max_speech_duration_s == FLT_MAX ?
                                                                                                                              std::string("FLT_MAX").c_str() : std::to_string(params.vad_max_speech_duration_s).c_str());
    fprintf(stderr, "  -vp N,     --vad-speech-pad-ms           N [%-7d] VAD speech padding (extend segments)\n",             params.vad_speech_pad_ms);
    fprintf(stderr, "  -vo N,     --vad-samples-overlap         N [%-7.2f] VAD samples overlap (seconds between segments)\n", params.vad_samples_overlap);
    fprintf(stderr, "  -pn NAME,  --person NAME        [%-7s] person name (for prompt selection)\n",             params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME      [%-7s] bot name (to display)\n",                          params.bot_name.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper      [%-7s] whisper model file\n",                             params.model_wsp.c_str());
    fprintf(stderr, "  -mll FILE, --model-llama       [%-7s] llama model file\n",                               params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT         [%-7s] command for TTS\n",                                params.speak.c_str());
    fprintf(stderr, "  -sf FILE, --speak-file         [%-7s] file to pass to TTS\n",                            params.speak_file.c_str());
    fprintf(stderr, "  --prompt-file FNAME            [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "\n");
}

const std::string k_prompt_llama = R"(Text transcript of a never ending dialog, where {0} interacts with an AI assistant named {1}.
{1} is helpful, kind, honest, friendly, good at writing and never fails to answer {0}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {0} and {1} say aloud to each other.
The transcript only includes text, it does not include markup like HTML and Markdown.
{1} responds with short and concise answers.

{0}{4} Hello, {1}!
{1}{4} Hello {0}! How may I help you today?
{0}{4} What time is it?
{1}{4} It is {2} o'clock.
{0}{4} What year is it?
{1}{4} We are in {3}.
{0}{4} What is a cat?
{1}{4} A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{0}{4} Name a color.
{1}{4} Blue
{0}{4})";

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
        printf("error: cannot use both --diarize and --tinydiarize\n");
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
 
    if (!ctx_wsp) {
        printf("No whisper.cpp model specified. Please provide using -mw <modelfile>\n");
        return 3;
    }

    // initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
    whisper_ctx_init_openvino_encoder(ctx_wsp, nullptr, params.openvino_encode_device.c_str(), nullptr);

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
            printf("error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
            return 4;
        } else {
            printf("%s: grammar:\n", __func__);
            grammar_parser::print_grammar(stderr, grammar);
            printf("\n");
        }
    }

    if (!whisper_is_multilingual(ctx_wsp)) {
        if (params.language != "en" || params.translate) {
            params.language = "en";
            params.translate = false;
            printf("%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
        }
    }

    if (params.detect_language) {
        params.language = "auto";
    }

    if (!params.no_prints) {
        // print system information
        printf("\n");
        printf("system_info: n_threads = %d / %d | %s\n",
                params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());

        // print some info about the processing
        printf("\n");
        printf("%s: processing -> %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                __func__, params.n_threads, params.n_processors, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.tinydiarize ? "tdrz = 1, " : "",
                params.no_timestamps ? 0 : 1);

        if (params.print_colors) {
            printf("%s: color scheme: red (low confidence), yellow (medium), green (high confidence)\n", __func__);
        } else if (params.print_confidence) {
            printf("%s: confidence: highlighted (low confidence), underlined (medium), dim (high confidence)\n", __func__);
        }
        printf("\n");
    }

    // llama init
    llama_backend_init();

    auto lmparams = llama_model_default_params();
    if (!params.use_gpu) {
        lmparams.n_gpu_layers = 0;
    } else {
        lmparams.n_gpu_layers = params.n_gpu_layers;
    }

    struct llama_model * model_llama = llama_model_load_from_file(params.model_llama.c_str(), lmparams);
    if (!model_llama) {
        printf("No llama.cpp model specified. Please provide using -ml <modelfile>\n");
        return 1;
    }

    const llama_vocab * vocab_llama = llama_model_get_vocab(model_llama);

    llama_context_params lcparams = llama_context_default_params();

    // tune these to your liking
    lcparams.n_ctx      = 2048;
    lcparams.n_threads  = params.n_threads;
    lcparams.flash_attn = params.flash_attn;

    struct llama_context * ctx_llama = llama_init_from_model(model_llama, lcparams);

    // print some info about the processing
    {
        printf("\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                printf("%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        printf("%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        printf("\n");
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
                printf("%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
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
            printf("%s: audio.init() failed!\n", __func__);
            return 1;
        }
        
        audio.resume();

        bool is_running = true;
        int n_tokens  = 0;
        const std::string chat_symb = ":";

        // construct the initial prompt for LLaMA inference
        std::string prompt_llama = params.prompt.empty() ? k_prompt_llama : params.prompt;

        // need to have leading ' '
        prompt_llama.insert(0, 1, ' ');

        prompt_llama = ::replace(prompt_llama, "{0}", params.person);
        prompt_llama = ::replace(prompt_llama, "{1}", params.bot_name);
        {
            // get time string
            std::string time_str;
            {
                time_t t = time(0);
                struct tm * now = localtime(&t);
                char buf[128];
                strftime(buf, sizeof(buf), "%H:%M", now);
                time_str = buf;
            }
            prompt_llama = ::replace(prompt_llama, "{2}", time_str);
        }

        {
            // get year string
            std::string year_str;
            {
                time_t t = time(0);
                struct tm * now = localtime(&t);
                char buf[128];
                strftime(buf, sizeof(buf), "%Y", now);
                year_str = buf;
            }
            prompt_llama = ::replace(prompt_llama, "{3}", year_str);
        }

        prompt_llama = ::replace(prompt_llama, "{4}", chat_symb);

        llama_batch batch = llama_batch_init(llama_n_ctx(ctx_llama), 0, 1);

        // init sampler
        const float top_k = 5;
        const float top_p = 0.80f;
        const float temp  = 0.30f;

        const int seed = 0;

        auto sparams = llama_sampler_chain_default_params();

        llama_sampler * smpl = llama_sampler_chain_init(sparams);

        if (temp > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp (temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist (seed));
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        }

        // init session
        std::string path_session = params.path_session;
        std::vector<llama_token> session_tokens;
        auto embd_inp = ::llama_tokenize(ctx_llama, prompt_llama, true);

        if (!path_session.empty()) {
            printf("%s: attempting to load saved session from %s\n", __func__, path_session.c_str());

            // fopen to check for existing session
            FILE * fp = std::fopen(path_session.c_str(), "rb");
            if (fp != NULL) {
                std::fclose(fp);

                session_tokens.resize(llama_n_ctx(ctx_llama));
                size_t n_token_count_out = 0;
                if (!llama_state_load_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                    printf("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                    return 1;
                }
                session_tokens.resize(n_token_count_out);
                for (size_t i = 0; i < session_tokens.size(); i++) {
                    embd_inp[i] = session_tokens[i];
                }

                printf("%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
            } else {
                printf("%s: session file does not exist, will create\n", __func__);
            }
        }

        // evaluate the initial prompt
        printf("\n%s : initializing - please wait ...\n", __func__);

        // prepare batch
        {
            batch.n_tokens = embd_inp.size();

            for (int i = 0; i < batch.n_tokens; i++) {
                batch.token[i]     = embd_inp[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = i == batch.n_tokens - 1;
            }
        }

        if (llama_decode(ctx_llama, batch)) {
            printf("%s : failed to decode\n", __func__);
            return 1;
        }

        if (params.verbose_prompt) {
            printf("\n%s", prompt_llama.c_str());
        }

        // debug message about similarity of saved session, if applicable
        size_t n_matching_session_tokens = 0;
        if (session_tokens.size()) {
            for (llama_token id : session_tokens) {
                if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                    break;
                }
                n_matching_session_tokens++;
            }
            if (n_matching_session_tokens >= embd_inp.size()) {
                printf("%s: session file has exact match for prompt!\n", __func__);
            } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
                printf("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            } else {
                printf("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            }
        }

        // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
        // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
        // initial prompt so it doesn't need to be an exact match.
        bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < (embd_inp.size() * 3 / 4);

        // text inference variables
        const int voice_id = 2;
        const int n_keep   = embd_inp.size();
        const int n_ctx    = llama_n_ctx(ctx_llama);

        char buffer[5] = {0};
        char tpBuffer[5] = {0};

        int n_past = n_keep;
        int n_prev = 64; // TODO arg
        int n_session_consumed = !path_session.empty() && session_tokens.size() > 0 ? session_tokens.size() : 0;

        std::vector<llama_token> embd;

        // reverse prompts for detecting when it's time to stop speaking
        std::vector<std::string> antiprompts = {
            params.person + chat_symb,
        };

        printf("Please start speech-to-text with %s.\n", params.bot_name.c_str());
        printf("%s: done! start speaking in the microphone.\n", params.bot_name.c_str());
        printf("%s%s ", params.person.c_str(), chat_symb.c_str());

        // wait for 3 second to avoid any buffered noise
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        audio.clear();

        while (is_running)
        {
            // handle Ctrl + C
            is_running = sdl_poll_events();

            // delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            audio.get(1500, pcmf32);

            if (::vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy)) {
                // fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                // we have heard the activation phrase, now detect the commands
                audio.get(params.command_ms, pcmf32);

                // const auto t_start = std::chrono::high_resolution_clock::now();
                std::string result = "";
                if (whisper_full_parallel(ctx_wsp, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) == 0) {
                    const int n_segments = whisper_full_n_segments(ctx_wsp);

                    for (int i = 0; i < n_segments; ++i) {
                        const char * text = whisper_full_get_segment_text(ctx_wsp, i);
    
                        result += text;
                    }
                }

                // remove text between brackets using regex
                {
                    std::regex re("\\[.*?\\]");
                    result = std::regex_replace(result, re, "");
                }

                // remove text between brackets using regex
                {
                    std::regex re("\\(.*?\\)");
                    result = std::regex_replace(result, re, "");
                }

                // remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '
                result = std::regex_replace(result, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");

                // take first line
                result = result.substr(0, result.find_first_of('\n'));

                // remove leading and trailing whitespace
                result = std::regex_replace(result, std::regex("^\\s+"), "");
                result = std::regex_replace(result, std::regex("\\s+$"), "");

                const std::vector<llama_token> tokens = llama_tokenize(ctx_llama, result.c_str(), false);

                if (result.empty() || tokens.empty()) {
                    audio.clear();
                    continue;
                }

                while(fscanf(stdin, "%s\n", tpBuffer) > 0) {
                    scanf("%s\n", tpBuffer);
                    strcmp(buffer, tpBuffer);
                    memset(tpBuffer, 0, sizeof(tpBuffer));
                    std::string strIsOnline(buffer);
                    
                    if (strIsOnline == "OFF") {
                        fprintf(stdout, "network offline: whisper\n");
                    } else if (strIsOnline == "ON") {
                        fprintf(stdout, "network online: whisper\n");
                    }
                }

                // if(!strcmp(buffer, "ON")) {
                //     result.insert(0, 1, ' ');
                //     result += "\n" + params.person + chat_symb;
                //     printf("%s%s%s", "\033[1m", result.c_str(), "\033[0m");
                //     audio.clear();
                //     continue;
                // } else {
                    result.insert(0, 1, ' ');
                    result += "\n" + params.bot_name + chat_symb;
                    printf("%s%s%s", "\033[1m", result.c_str(), "\033[0m");
                // }

                embd = ::llama_tokenize(ctx_llama, result, false);

                // Append the new input tokens to the session_tokens vector
                if (!path_session.empty()) {
                    session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
                }

                // text inference
                bool done = false;
                std::string text_to_speak;
                while (true) {
                    // predict
                    if (embd.size() > 0) {
                        if (n_past + (int) embd.size() > n_ctx) {
                            n_past = n_keep;

                            // insert n_left/2 tokens at the start of embd from last_n_tokens
                            embd.insert(embd.begin(), embd_inp.begin() + embd_inp.size() - n_prev, embd_inp.end());
                            // stop saving session if we run out of context
                            path_session = "";
                        }

                        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                        // REVIEW
                        if (n_session_consumed < (int) session_tokens.size()) {
                            size_t i = 0;
                            for ( ; i < embd.size(); i++) {
                                if (embd[i] != session_tokens[n_session_consumed]) {
                                    session_tokens.resize(n_session_consumed);
                                    break;
                                }

                                n_past++;
                                n_session_consumed++;

                                if (n_session_consumed >= (int) session_tokens.size()) {
                                    i++;
                                    break;
                                }
                            }
                            if (i > 0) {
                                embd.erase(embd.begin(), embd.begin() + i);
                            }
                        }

                        if (embd.size() > 0 && !path_session.empty()) {
                            session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                            n_session_consumed = session_tokens.size();
                        }

                        // prepare batch
                        {
                            batch.n_tokens = embd.size();

                            for (int i = 0; i < batch.n_tokens; i++) {
                                batch.token[i]     = embd[i];
                                batch.pos[i]       = n_past + i;
                                batch.n_seq_id[i]  = 1;
                                batch.seq_id[i][0] = 0;
                                batch.logits[i]    = i == batch.n_tokens - 1;
                            }
                        }

                        if (llama_decode(ctx_llama, batch)) {
                            printf("%s : failed to decode\n", __func__);
                            return 1;
                        }
                    }

                    embd_inp.insert(embd_inp.end(), embd.begin(), embd.end());
                    n_past += embd.size();

                    embd.clear();

                    if (done) break;

                    {
                        // out of user input, sample next token
                        if (!path_session.empty() && need_to_save_session) {
                            need_to_save_session = false;
                            llama_state_save_file(ctx_llama, path_session.c_str(), session_tokens.data(), session_tokens.size());
                        }

                        const llama_token id = llama_sampler_sample(smpl, ctx_llama, -1);

                        if (id != llama_vocab_eos(vocab_llama)) {
                            // add it to the context
                            embd.push_back(id);

                            text_to_speak += llama_token_to_piece(ctx_llama, id);

                            printf("%s", llama_token_to_piece(ctx_llama, id).c_str());
                        }
                    }

                    {
                        std::string last_output;
                        for (int i = embd_inp.size() - 16; i < (int) embd_inp.size(); i++) {
                            last_output += llama_token_to_piece(ctx_llama, embd_inp[i]);
                        }
                        last_output += llama_token_to_piece(ctx_llama, embd[0]);

                        for (std::string & antiprompt : antiprompts) {
                            if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                                done = true;
                                text_to_speak = ::replace(text_to_speak, antiprompt, "");
                                fflush(stdout);
                                need_to_save_session = true;
                                break;
                            }
                        }
                    }

                    is_running = sdl_poll_events();

                    if (!is_running) {
                        break;
                    }
                }

                speak_with_file(params.speak, text_to_speak, params.speak_file, voice_id);

                audio.clear();
            }
        }

        audio.pause();

        whisper_print_timings(ctx_wsp);
        whisper_free(ctx_wsp);

        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx_llama);

        llama_sampler_free(smpl);
        llama_batch_free(batch);
        llama_free(ctx_llama);

        llama_backend_free();
    }

    return 0;
}
