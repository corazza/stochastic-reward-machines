#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include <stdlib.h>
#include <stdio.h>

#include "json.hpp"
#include "z3++.h"

using namespace z3;
using json = nlohmann::json;

typedef std::pair<std::vector<std::string>, std::vector<float> > Trace;
typedef std::pair<std::vector<std::string>, std::vector<float> > PrefixTrace;
typedef std::vector<std::string> Language;
// empty_transition : string
typedef std::pair<std::pair<int, std::string>, int> DDictKey;
typedef std::pair<int, std::string> ODictKey;
typedef std::string XDictKey;

const int INITIAL_STATE = 0;
const int TERMINAL_STATE = -1;

struct HashableTrace {
    Trace trace;
    std::string prefix;
    std::string lm;
    bool terminal;
};

std::vector<Trace> extract_container(json from) {
    std::vector<Trace> X;

    for (auto& el : from) {
        std::vector<std::string> labels;
        std::vector<float> rewards;

        for (auto &l : el[0]) {
            labels.push_back(l);
        }

        for (auto &r : el[1]) {
            rewards.push_back(r);
        }

        auto trace = std::make_pair(labels, rewards);

        X.push_back(trace);
    }

    return X;
}

DDictKey d_dict_key(int p, const std::string& a, int q) {
    auto inner = std::make_pair(p, std::string(a));
    return std::make_pair(inner, q);
}

ODictKey o_dict_key(int p, const std::string& a) {
    return std::make_pair(p, a);
}

XDictKey x_dict_key(const std::string &labels, int p) {
    return labels + std::to_string(p);
}

expr add_x(const std::string &labels, int p, context *c, std::unordered_map<XDictKey, expr> *x_dict, int *counter) {
    XDictKey key = x_dict_key(labels, p);
    auto queried = x_dict->find(key);
    if (queried != x_dict->end()) {
        return queried->second;
    }

    expr x = c->bool_const(std::to_string(*counter).c_str());
    *counter += 1;
    x_dict->emplace(key, x);
    return x;
}

expr build_disj(std::vector<expr> es) {
    expr acc = es[0];
    for (int i = 0; i < es.size(); ++i) {
        acc = acc || es[i];
    }
    return acc;
}

expr build_conj(std::vector<expr> es) {
    expr acc = es[0];
    for (int i = 0; i < es.size(); ++i) {
        acc = acc && es[i];
    }
    return acc;
}

void print_vec_strings(const std::vector<std::string> &vec) {
    std::cout << "[ ";
    for (auto const& s : vec) {
        std::cout << s << " , ";
    }
    std::cout << " ]";
}

void print_vec_floats(const std::vector<float> &vec) {
    std::cout << "[ ";
    for (auto const& s : vec) {
        std::cout << s << " , ";
    }
    std::cout << " ]";
}

void log(int argc, const char *asdf) {
    if (argc > 2) {
        std::cout << asdf << std::endl;
    }
}

std::vector<HashableTrace> prefixes(std::vector<Trace>& X) {
    std::vector<HashableTrace> result;

    for (auto const& trace : X) {
        std::ostringstream prefix;

        for (int i = 0; i < trace.first.size(); ++i) {
            std::string lm = prefix.str();

            prefix << trace.first[i] << "|";
            std::vector<std::string>::const_iterator l_first = trace.first.begin() + 0;
            std::vector<std::string>::const_iterator l_last = trace.first.begin() + i + 1;
            std::vector<std::string> labels(l_first, l_last);


            std::vector<float>::const_iterator r_first = trace.second.begin() + 0;
            std::vector<float>::const_iterator r_last = trace.second.begin() + i + 1;
            std::vector<float> rewards(r_first, r_last);

            HashableTrace h_trace;
            h_trace.trace = std::make_pair(labels, rewards);
            h_trace.prefix = prefix.str();
            h_trace.lm = lm;
            h_trace.terminal = ((i + 1) == trace.first.size());

            result.push_back(h_trace);
        }
    }

    return result;
}


bool smt_noise(int argc, float epsilon_f, std::vector<Trace> X, std::vector<Trace> X_tl, int n_states, bool infer_termination, Language language, std::string empty_transition, json *results) {
    std::vector<int> states;
    for (int i = 0; i < n_states; ++i) {
        states.push_back(i);
    }
    states.push_back(TERMINAL_STATE);

    std::vector<HashableTrace> x_prefixes = prefixes(X);

    std::string epsilon_s = std::to_string(epsilon_f);
    std::string empty_label_sequence("");

    context c;
    solver s(c);
    int var_counter = 0;

    expr epsilon = c.real_val(epsilon_s.c_str());

    std::map<DDictKey, expr> d_dict;
    std::map<ODictKey, expr> o_dict;
    std::unordered_map<XDictKey, expr> x_dict;
    
    for (const auto& p : states) {
        for (const auto& a : language) {
            auto o_key = o_dict_key(p, a);
            expr o = c.real_const(std::to_string(var_counter++).c_str());
            o_dict.emplace(o_key, o);
            for (const auto& q : states) {
                auto d_key = d_dict_key(p, a, q);
                expr d = c.bool_const(std::to_string(var_counter++).c_str());
                d_dict.emplace(d_key, d);
            }
        }
    }

    log(argc, "generating constraints");
    for (const auto& p : states) {
        for (const auto& l : language) {
            std::vector<expr> to_disj;
            for (const auto& q : states) {
                auto key = d_dict_key(p, l, q);
                expr d = d_dict.at(key);
                to_disj.push_back(d);
            }
            s.add(build_disj(to_disj));
            for (const auto& q1: states) {
                for (const auto& q2 : states) {
                    if (q1 == q2) {
                        continue;
                    }
                    auto p_l_q1 = d_dict.at(d_dict_key(p, l, q1));
                    auto p_l_q2 = d_dict.at(d_dict_key(p, l, q2));
                    s.add((!p_l_q1) || (!p_l_q2));
                }
            }
        }
    }

    s.add(add_x(empty_label_sequence, INITIAL_STATE, &c, &x_dict, &var_counter));
    for (auto const& p : states) {
        if (p == INITIAL_STATE) {
            continue;
        }
        s.add(!add_x(empty_label_sequence, p, &c, &x_dict, &var_counter));
    }

    for (auto const& h_trace : x_prefixes) {
        if (h_trace.trace.first.size() == 0) {
            continue;
        }

        // std::vector<std::string> lm(h_trace.trace.first.begin(), h_trace.trace.first.begin()+h_trace.trace.first.size()-1);
        std::string l = h_trace.trace.first.back();
        std::string r = std::to_string(h_trace.trace.second.back());

        for (auto const& p : states) {
            expr x_1 = add_x(h_trace.lm, p, &c, &x_dict, &var_counter);
            expr o = o_dict.at(o_dict_key(p, l));
            s.add(implies(x_1, ((c.real_val(r.c_str()) - o) >= -epsilon) && ((c.real_val(r.c_str()) - o) <= epsilon)));

            for (auto const& q : states) {
                expr d = d_dict.at(d_dict_key(p, l, q));
                expr x_2 = add_x(h_trace.prefix, q, &c, &x_dict, &var_counter);
                s.add(implies(x_1 && d, x_2));
            }
        }
    }

    if (infer_termination) {
        for (auto const& h_trace : x_prefixes) {
            if (h_trace.trace.first.size() == 0) {
                continue;
            }

            // std::vector<std::string> lm(h_trace.trace.first.begin(), h_trace.trace.first.begin()+h_trace.trace.first.size()-1);
            std::string l = h_trace.trace.first.back();
            std::string r = std::to_string(h_trace.trace.second.back());

            for (auto const& p : states) {
                if (p == TERMINAL_STATE) {
                    continue;
                }
                expr x_1 = add_x(h_trace.lm, p, &c, &x_dict, &var_counter);
                expr d = d_dict.at(d_dict_key(p, l, TERMINAL_STATE));
                if (!h_trace.terminal) {
                    s.add(implies(x_1, !d));
                } else {
                    auto found = std::find(X_tl.begin(), X_tl.end(), h_trace.trace);
                    expr d_t = found != X_tl.end() ? !d : d;
                    s.add(implies(x_1, d_t));
                }
            }
        }

        for (auto const& p : states) {
            for (auto const& l : language) {
                if (p != TERMINAL_STATE) {
                    expr d = d_dict.at(d_dict_key(TERMINAL_STATE, l, p));
                    s.add(!d);
                } 
                expr o = o_dict.at(o_dict_key(TERMINAL_STATE, l));
                s.add(o == 0);
            }
        }
    } else {
        for (auto const& p : states) {
            for (auto const& l : language) {
                expr d = d_dict.at(d_dict_key(p, l, TERMINAL_STATE));
                s.add(!d);
            }
        }
    }

    log(argc, "checking");

    auto result = s.check();
    
    if (argc > 2) {
        std::cout << result << std::endl;
    }

    if (result != sat) {
        return false;
    }

    log(argc, "getting model");
    model m = s.get_model();
    log(argc, "got model");
    
    json transitions;

    // circumventing Z3 outputs for non-noisy case
    std::set<float> all_outputs_set;
    if (epsilon_f == 0) {
        for (auto const& trace : X) {
            for (auto const& reward : trace.second) {
                all_outputs_set.insert(reward);
            }
        }
    }
    std::vector<float> all_outputs(all_outputs_set.begin(), all_outputs_set.end());

    for (auto const& [key, val] : d_dict) {
        if (m.eval(val).is_true()) {
            auto o_key = o_dict_key(key.first.first, key.first.second);
            auto o = m.eval(o_dict.at(o_key));
            double out = 0.0;
            if (o.is_numeral()){
                out = (double) o.numerator().get_numeral_int64() / (double) o.denominator().get_numeral_int64();
                if (epsilon_f == 0) {
                    double min = abs(out - all_outputs[0]);
                    double min_output = all_outputs[0];
                    for (auto reward : all_outputs) {
                        double distance = abs(out - reward);
                        if (distance < min) {
                            min = distance;
                            min_output = reward;
                        }
                    }
                    out = min_output;
                }
            }
            json transition;
            transition.push_back(key.first.first);
            transition.push_back(key.first.second);
            transition.push_back(key.second);
            transition.push_back(std::to_string(out));
            transitions.push_back(transition);
        }
    }

    *results = transitions;
    log(argc, "made transitions");

    return true;
}

int main(int argc, char *argv[]) {
    log(argc, "SMT executable started");

    if (argc > 2) {
        std::cout << "reading from " << argv[1] << std::endl;
    }
    
    std::ifstream i(argv[1]);
    json j;
    i >> j;
    i.close();

    std::vector<Trace> X = extract_container(j["X"]);
    std::vector<Trace> X_tl = extract_container(j["X_tl"]);
    Language language;

    for (auto& el : j["language"]) {
        language.push_back(el);
    }

    int seed = j["seed"];
    set_param("sat.random_seed", seed);
    set_param("smt.random_seed", seed);

    json transitions;
    bool result = smt_noise(argc, j["epsilon"], X, X_tl, j["n_states"], j["infer_termination"], language, j["empty_transition"], &transitions);

    std::ofstream o(argv[1]);
    if (result) {
        o << transitions;
    } else {
        o << "[\"unsat\"]";
    }
    o.close();
}
