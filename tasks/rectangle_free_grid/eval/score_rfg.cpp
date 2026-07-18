// Standalone faithful port of the FrontierCS chk.cc check() + computeU() + scoring,
// WITHOUT testlib (so it is self-contained and portable to any Linux/macOS box).
// Verified to match the upstream chk.cc exactly (ratio 0.5915151515 on the 100x100
// sample). Reads: argv[1]=input file "n m", argv[2]=solution output "k\n r c\n...".
// Prints one JSON line: {"valid":bool,"k":int,"U":int,"ratio":float,"unbounded":float,"reason":"..."}
//   ratio     = min(k/(1.5*U), 1)        (== per-case score, chk.cc lines 176-180)
//   unbounded = max(0, k/(1.5*U))        (== per-case scoreRatioUnbounded)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
typedef long long LL;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: score_rfg <input> <output>\n"); return 2; }
    FILE* fin = fopen(argv[1], "r");
    FILE* fout = fopen(argv[2], "r");
    if (!fin || !fout) { printf("{\"valid\":false,\"reason\":\"cannot open files\"}\n"); return 2; }

    LL N, M;
    if (fscanf(fin, "%lld %lld", &N, &M) != 2) { printf("{\"valid\":false,\"reason\":\"bad input\"}\n"); return 2; }
    fclose(fin);

    long long kll;
    if (fscanf(fout, "%lld", &kll) != 1) { printf("{\"valid\":false,\"k\":0,\"reason\":\"no k\"}\n"); return 0; }
    LL kmax = (N > 0 && M > 0) ? std::min<LL>(N * M, 1000000000LL) : 0;
    if (kll < 0 || kll > kmax) { printf("{\"valid\":false,\"k\":%lld,\"reason\":\"k out of range\"}\n", kll); return 0; }
    int K = (int)kll;
    vector<pair<int,int>> pts(K);
    for (int i = 0; i < K; i++) {
        long long r, c;
        if (fscanf(fout, "%lld %lld", &r, &c) != 2) { printf("{\"valid\":false,\"k\":%d,\"reason\":\"truncated output at %d\"}\n", K, i); return 0; }
        if (r < 1 || r > N || c < 1 || c > M) { printf("{\"valid\":false,\"k\":%d,\"reason\":\"coord out of range\"}\n", K); return 0; }
        pts[i] = {(int)r, (int)c};
    }
    fclose(fout);

    string reason = "ok";
    bool ok = true;

    // Duplicate check via encoded ids (mirror chk.cc)
    {
        vector<unsigned long long> enc; enc.reserve(K);
        for (int i = 0; i < K; i++) {
            unsigned long long id = (unsigned long long)(pts[i].first - 1) * (unsigned long long)M + (unsigned long long)(pts[i].second - 1);
            enc.push_back(id);
        }
        sort(enc.begin(), enc.end());
        for (int i = 1; i < K; i++) if (enc[i] == enc[i-1]) { ok = false; reason = "duplicate coordinates"; break; }
    }

    if (ok && K > 0 && N > 1 && M > 1) {
        // Rectangle check: no two rows share >=2 columns (heavy-light, mirror chk.cc).
        vector<vector<int>> rows((size_t)N + 1), cols((size_t)M + 1);
        for (auto& p : pts) { rows[p.first].push_back(p.second); cols[p.second].push_back(p.first); }
        for (int r = 1; r <= N; r++) if (!rows[r].empty()) sort(rows[r].begin(), rows[r].end());

        const int B = 300;
        vector<int> heavyRows;
        for (int r = 1; r <= N; r++) if ((int)rows[r].size() > B) heavyRows.push_back(r);
        if (ok && !heavyRows.empty()) {
            vector<int> cnt((size_t)N + 1, 0), touched;
            for (int r : heavyRows) {
                for (int c : rows[r]) for (int rr : cols[c]) {
                    if (rr == r) continue;
                    if (cnt[rr] == 0) touched.push_back(rr);
                    cnt[rr]++;
                    if (cnt[rr] >= 2) { ok = false; reason = "rectangle (heavy rows share 2 cols)"; break; }
                }
                if (!ok) break;
                for (int rr : touched) cnt[rr] = 0;
                touched.clear();
            }
        }
        if (ok) {
            unordered_map<uint64_t, bool> seen;
            for (int r = 1; r <= N && ok; r++) {
                int d = (int)rows[r].size();
                if (d < 2 || d > B) continue;
                auto& v = rows[r];
                for (int i = 0; i < d && ok; i++) for (int j = i + 1; j < d; j++) {
                    uint64_t key = ((uint64_t)(uint32_t)v[i] << 32) | (uint32_t)v[j];
                    if (seen.count(key)) { ok = false; reason = "rectangle (column pair in two rows)"; break; }
                    seen[key] = true;
                }
            }
        }
    }

    if (!ok) { printf("{\"valid\":false,\"k\":%d,\"reason\":\"%s\"}\n", K, reason.c_str()); return 0; }

    // computeU (mirror chk.cc)
    long double sn = sqrtl((long double)N), sm = sqrtl((long double)M);
    long double v1 = floorl((long double)N * sm + (long double)M);
    long double v2 = floorl((long double)M * sn + (long double)N);
    long double v3 = (long double)N * (long double)M;
    long double U = min(v1, min(v2, v3));
    if (U < 0) U = 0;
    LL Ui = (LL)U;

    double ratio, unbounded;
    if (Ui <= 0) { ratio = (K > 0 ? 1.0 : 0.0); unbounded = ratio; }
    else {
        long double rl = (long double)K / ((long double)Ui * 1.5L);
        unbounded = max(0.0, (double)rl);
        long double cl = rl; if (cl > 1.0L) cl = 1.0L; if (cl < 0.0L) cl = 0.0L;
        ratio = (double)cl;
    }
    printf("{\"valid\":true,\"k\":%d,\"U\":%lld,\"ratio\":%.6f,\"unbounded\":%.6f,\"reason\":\"ok\"}\n", K, Ui, ratio, unbounded);
    return 0;
}
