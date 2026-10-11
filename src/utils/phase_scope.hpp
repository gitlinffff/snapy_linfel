#pragma once
// Profiling only (chengcli/snapy#291): sequential torch-profiler phase scopes.
// PhaseScope ps("a"); ...; ps.start("b"); ... closes "a" and opens "b".
#include <ATen/record_function.h>

#include <memory>

namespace snap {

class PhaseScope {
 public:
  explicit PhaseScope(const char* name) { start(name); }
  ~PhaseScope() { stop(); }
  void start(const char* name) {
    stop();
    guard_ = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (guard_->isActive()) guard_->before(name);
  }
  void stop() { guard_.reset(); }

 private:
  std::unique_ptr<at::RecordFunction> guard_;
};

}  // namespace snap
