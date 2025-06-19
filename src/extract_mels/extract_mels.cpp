#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <filesystem>

#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include "cnpy.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#ifdef __cplusplus
}
#endif

namespace fs = std::filesystem;

struct Args {
    std::string input_jsonl;
    std::string output_dir;
    std::string index_csv;
    std::string config = "config.yaml";
    int sr = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 80;
    bool preemphasis = false;
    bool no_norm = false;
};


std::vector<float> parse_audio(const std::string &path, int target_sr) {
    AVFormatContext *fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("Unable to open audio: " + path);
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not find stream info: " + path);
    }

    const AVCodec *dec = nullptr;
    int stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
    if (stream_index < 0) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("No audio stream found in: " + path);
    }

    AVStream *stream = fmt_ctx->streams[stream_index];
    AVCodecContext *dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not allocate codec context");
    }
    if (avcodec_parameters_to_context(dec_ctx, stream->codecpar) < 0) {
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not copy codec parameters");
    }
    if (avcodec_open2(dec_ctx, dec, nullptr) < 0) {
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not open decoder");
    }

    int64_t in_layout = dec_ctx->channel_layout;
    if (!in_layout) in_layout = av_get_default_channel_layout(dec_ctx->channels);
    SwrContext *swr = swr_alloc_set_opts(nullptr,
        AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, target_sr,
        in_layout, dec_ctx->sample_fmt, dec_ctx->sample_rate,
        0, nullptr);
    if (!swr || swr_init(swr) < 0) {
        if (swr) swr_free(&swr);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not initialize resampler");
    }

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    std::vector<float> data;

    auto decode = [&](AVPacket *packet) {
        if (avcodec_send_packet(dec_ctx, packet) < 0) return;
        while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
            int out_samples = av_rescale_rnd(swr_get_delay(swr, dec_ctx->sample_rate) + frame->nb_samples,
                                            target_sr, dec_ctx->sample_rate, AV_ROUND_UP);
            std::vector<float> out_buf(out_samples);
            uint8_t *out_ptr = reinterpret_cast<uint8_t*>(out_buf.data());
            int ret = swr_convert(swr, &out_ptr, out_samples,
                                  (const uint8_t**)frame->extended_data, frame->nb_samples);
            if (ret < 0) continue;
            out_buf.resize(ret);
            data.insert(data.end(), out_buf.begin(), out_buf.end());
        }
    };

    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_index) decode(pkt);
        av_packet_unref(pkt);
    }
    decode(nullptr); // flush

    av_frame_free(&frame);
    av_packet_free(&pkt);
    swr_free(&swr);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    return data;
}

std::vector<float> preemphasis(const std::vector<float> &signal, float coef) {
    if (signal.empty()) return {};
    std::vector<float> out(signal.size());
    out[0] = signal[0];
    for (size_t i = 1; i < signal.size(); ++i) {
        out[i] = signal[i] - coef * signal[i - 1];
    }
    return out;
}

std::vector<float> trim_silence(const std::vector<float> &signal, float top_db) {
    if (signal.empty()) return {};
    float max_amp = 0.0f;
    for (float v : signal) max_amp = std::max(max_amp, std::abs(v));
    if (max_amp == 0.0f) return signal;

    float thresh = max_amp * std::pow(10.0f, -top_db / 20.0f);
    size_t start = 0;
    while (start < signal.size() && std::abs(signal[start]) < thresh) ++start;
    size_t end = signal.size();
    while (end > start && std::abs(signal[end - 1]) < thresh) --end;
    return std::vector<float>(signal.begin() + start, signal.begin() + end);
}

// naive DFT for demonstration
std::vector<std::vector<float>> stft(const std::vector<float> &y, int n_fft, int hop) {
    size_t frames = y.size() / hop + 1;
    std::vector<std::vector<float>> spec(frames, std::vector<float>(n_fft/2 + 1));
    std::vector<std::complex<float>> X(n_fft);
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f - 0.5f * std::cos(2 * M_PI * i / (n_fft-1));
    }

    for (size_t f = 0; f < frames; ++f) {
        size_t start = f * hop;
        for (int k = 0; k < n_fft; ++k) {
            float val = 0.0f;
            if (start + k < y.size()) val = y[start + k];
            X[k] = std::complex<float>(val * window[k], 0.0f);
        }
        // DFT
        for (int m = 0; m <= n_fft/2; ++m) {
            std::complex<float> sum(0.0f, 0.0f);
            for (int n = 0; n < n_fft; ++n) {
                float phi = -2.0f * M_PI * m * n / n_fft;
                std::complex<float> w(std::cos(phi), std::sin(phi));
                sum += X[n] * w;
            }
            spec[f][m] = std::norm(sum);
        }
    }
    return spec;
}

std::vector<std::vector<float>> mel_filterbank(int n_fft, int n_mels, int sr) {
    // compute mel filterbank
    std::vector<std::vector<float>> fb(n_mels, std::vector<float>(n_fft/2 + 1));
    auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };
    float mel_low = hz_to_mel(0);
    float mel_high = hz_to_mel(sr / 2);
    std::vector<float> mel_points(n_mels + 2);
    for (int m = 0; m < n_mels + 2; ++m) {
        mel_points[m] = mel_to_hz(mel_low + (mel_high - mel_low) * m / (n_mels + 1));
    }
    std::vector<int> bins(n_mels + 2);
    for (int m = 0; m < n_mels + 2; ++m) {
        bins[m] = static_cast<int>(std::floor((n_fft + 1) * mel_points[m] / sr));
    }
    for (int m = 0; m < n_mels; ++m) {
        for (int k = bins[m]; k < bins[m+1]; ++k) {
            fb[m][k] = (k - bins[m]) / float(bins[m+1] - bins[m]);
        }
        for (int k = bins[m+1]; k < bins[m+2]; ++k) {
            fb[m][k] = (bins[m+2] - k) / float(bins[m+2] - bins[m+1]);
        }
    }
    return fb;
}

std::vector<std::vector<float>> apply_mel(const std::vector<std::vector<float>> &spec, const std::vector<std::vector<float>> &fb) {
    std::vector<std::vector<float>> mel(spec.size(), std::vector<float>(fb.size()));
    for (size_t t = 0; t < spec.size(); ++t) {
        for (size_t m = 0; m < fb.size(); ++m) {
            float sum = 0.0f;
            for (size_t k = 0; k < spec[t].size(); ++k) {
                sum += spec[t][k] * fb[m][k];
            }
            mel[t][m] = sum;
        }
    }
    return mel;
}

std::vector<std::vector<float>> melspectrogram(const std::vector<float> &y, int sr, int n_fft, int hop, int n_mels) {
    auto spec = stft(y, n_fft, hop);
    auto fb = mel_filterbank(n_fft, n_mels, sr);
    auto mel = apply_mel(spec, fb);
    return mel;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        auto next = [&]() -> std::string { if (i+1 < argc) return argv[++i]; throw std::runtime_error("Missing arg for " + opt); };
        if (opt == "--input_jsonl") args.input_jsonl = next();
        else if (opt == "--output_dir") args.output_dir = next();
        else if (opt == "--index_csv") args.index_csv = next();
        else if (opt == "--config") args.config = next();
        else if (opt == "--sr") args.sr = std::stoi(next());
        else if (opt == "--n_fft") args.n_fft = std::stoi(next());
        else if (opt == "--hop_length") args.hop_length = std::stoi(next());
        else if (opt == "--n_mels") args.n_mels = std::stoi(next());
        else if (opt == "--preemphasis") args.preemphasis = true;
        else if (opt == "--no_norm") args.no_norm = true;
    }
    if (args.input_jsonl.empty() || args.output_dir.empty()) {
        throw std::runtime_error("--input_jsonl and --output_dir are required");
    }
    return args;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        YAML::Node config = YAML::LoadFile(args.config);
        int top_db = config["top_db"].as<int>(30);

        std::ifstream fin(args.input_jsonl);
        if (!fin.is_open()) throw std::runtime_error("Cannot open input jsonl");
        std::vector<std::string> index_lines;
        fs::create_directories(args.output_dir);

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            auto entry = nlohmann::json::parse(line);
            std::string audio_path = entry["audio_path"].get<std::string>();
            std::vector<std::string> phonemes = entry["phonemes"].get<std::vector<std::string>>();
            std::string uid = fs::path(audio_path).stem().string();

            auto y = parse_audio(audio_path, args.sr);
            y = trim_silence(y, static_cast<float>(top_db));
            if (args.preemphasis) {
                y = preemphasis(y, 0.97f);
            }
            auto mel = melspectrogram(y, args.sr, args.n_fft, args.hop_length, args.n_mels);

            // optional normalization
            if (!args.no_norm) {
                // convert power spectrum to decibel scale (ref=max)
                float max_val = 0.0f;
                for (const auto &row : mel) {
                    for (float v : row) {
                        if (v > max_val) max_val = v;
                    }
                }
                float ref = std::max(max_val, 1e-10f);
                float log_ref = 10.0f * std::log10(ref);
                for (auto &row : mel) {
                    for (float &v : row) {
                        float val = std::max(v, 1e-10f);
                        v = 10.0f * std::log10(val) - log_ref;
                    }
                }

                float mean = 0.0f; size_t count = 0;
                for (const auto &row : mel) for (float v : row) { mean += v; ++count; }
                mean /= count;
                float var = 0.0f;
                for (const auto &row : mel) for (float v : row) { float d = v - mean; var += d*d; }
                var /= count;
                float std = std::sqrt(var + 1e-6f);
                for (auto &row : mel) for (float &v : row) { v = (v - mean) / std; }
            }

            std::vector<int> shape = {(int)mel.size(), (int)mel[0].size()};
            std::vector<float> flat; flat.reserve(shape[0]*shape[1]);
            for (auto &row : mel) flat.insert(flat.end(), row.begin(), row.end());

            fs::path out_path = fs::path(args.output_dir) / (uid + ".npz");
            cnpy::npz_save(out_path.string(), "mel", flat.data(), {shape[0], shape[1]}, "w");
            cnpy::npz_save(out_path.string(), "phonemes", phonemes.data(), {phonemes.size()}, "a");
            cnpy::npz_save(out_path.string(), "audio_path", audio_path.c_str(), {1}, "a");

            index_lines.push_back(uid);
        }
        if (!args.index_csv.empty()) {
            std::ofstream fout(args.index_csv);
            for (auto &id : index_lines) fout << id << "\n";
        }
        return 0;
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

