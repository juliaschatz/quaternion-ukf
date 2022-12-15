#include <eigen3/Eigen/Cholesky>
#include <iostream>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include "ukf.h"

using namespace std::chrono;
using namespace std::chrono_literals;

void quaternionMean(std::vector<Eigen::Quaterniond> quats, std::vector<double> weights, Eigen::Quaterniond& res, std::vector<Eigen::Vector3d>& errorVecs) {
  // Pick the first quat as the guess, since for sigma points that is the old mean
  Eigen::Quaterniond qt_last = quats[0];
  std::vector<Eigen::Vector3d> error_vecs;
  // Do up to 100 iterations
  for (int i = 0; i < 100; ++i) {
    Eigen::Quaterniond qt_lastinv = qt_last.inverse();
    // Last sum error vector
    Eigen::Vector3d weighted_error_vector = Eigen::Vector3d::Zero();
    error_vecs.clear();
    // Loop through each quaternion, calculate the error, weight it, and add back
    for (int j = 0; j < quats.size(); ++j) {
      Eigen::Quaterniond q = quats[j];
      Eigen::AngleAxisd err_aa(q * qt_lastinv);
      Eigen::Vector3d error_vector = err_aa.axis() * err_aa.angle();
      error_vecs.push_back(error_vector);
      weighted_error_vector += error_vector * weights[j];
    }
    Eigen::AngleAxisd err_aa(weighted_error_vector.norm(), weighted_error_vector.normalized());
    Eigen::Quaterniond qt_new = err_aa * qt_last;

    double err_angle = qt_new.angularDistance(qt_last);
    // If we haven't changed that much, exit early
    if (err_angle < 1e-3) {
      qt_last = qt_new;
      break;
    }
  }

  res = qt_last;
  errorVecs = error_vecs;
}

Matrix<double, 6, 6> discreteWhiteNoise3d2ndOrder(const double dt, const double variance) {
  Matrix<double, 6, 6> Q = Matrix<double,6,6>::Zero();
  Q.block(0, 0, 3, 3) = Matrix3d::Identity() * 0.25 * pow(dt, 4);
  Q.block(0, 3, 3, 3) = Q.block(3, 0, 3, 3) = Matrix3d::Identity() * 0.5 * pow(dt, 3);
  Q.block(3, 3, 3, 3) = Matrix3d::Identity() * pow(dt, 2);
  return Q * variance;
}
Quaterniond zyx(const double z, const double y, const double x) {
  AngleAxisd zRot(z, Eigen::Vector3d::UnitZ());
  AngleAxisd yRot(y, Eigen::Vector3d::UnitY());
  AngleAxisd xRot(x, Eigen::Vector3d::UnitX());
  return Quaterniond(zRot * yRot * xRot);
}

/*
 * Operators, necessary for templated integration
 * These should be pretty straightforward
 */
const UKFState UKFState::operator+(UKFState other) {
  return UKFState(
    this->position + other.position,
    this->velocity + other.velocity,
    other.orientation * this->orientation,
    this->remoteRate + other.remoteRate,
    this->localRate + other.localRate
  );
}
const UKFState UKFState::operator-(UKFState other) {
  return *this + (-other);
}
const UKFState UKFState::operator-() {
  return UKFState(
    -this->position,
    -this->velocity,
    this->orientation.inverse(),
    -this->remoteRate,
    -this->localRate
  );
}
const UKFState UKFState::operator+(UKFStateDerivative other) {
  Vector3d total_w = other.body_rate_local + other.frame_rate;
  return UKFState(
    this->position + other.velocity_remote,
    this->velocity + other.acceleration_remote,
    // Rotate the orientation by a rate vector
    AngleAxisd(total_w.norm(), total_w.normalized()) * this->orientation,
    this->remoteRate,
    this->localRate // zero angular acceleration
  );
}
const UKFState UKFState::operator*(const double d) {
  return UKFState(
    this->position * d,
    this->velocity * d,
    this->orientation, // Scaling an orientation doesn't make much sense, and it's not used anyway, so ignore
    this->remoteRate * d,
    this->localRate * d
  );
}
const UKFStateDerivative UKFState::derivative() {
  UKFStateDerivative usd;
  usd.velocity_remote = this->velocity + this->remoteRate.cross(this->position);
  usd.acceleration_remote = -this->remoteRate.cross(this->remoteRate.cross(this->position)) - 2 * this->remoteRate.cross(this->velocity);
  usd.body_rate_local = this->localRate;
  usd.frame_rate = this->remoteRate;
  return usd;
}
const UKFState UKFState::integrate(const double dt) {
  return integrateRK4<UKFState, UKFStateDerivative>(*this, [] (UKFState x, double t) {
    return x.derivative();
  }, 0, dt);
}
UKFStateDerivative UKFStateDerivative::operator*(double d) {
  UKFStateDerivative usd;
  usd.acceleration_remote = this->acceleration_remote * d;
  usd.body_rate_local = this->body_rate_local * d;
  usd.frame_rate = this->frame_rate * d;
  usd.velocity_remote = this->velocity_remote * d;
  return usd;
}
UKFStateDerivative UKFStateDerivative::operator+(UKFStateDerivative other) {
  UKFStateDerivative usd;
  usd.acceleration_remote = this->acceleration_remote + other.acceleration_remote;
  usd.body_rate_local = this->body_rate_local + other.body_rate_local;
  usd.frame_rate = this->frame_rate + other.frame_rate;
  usd.velocity_remote = this->velocity_remote + other.velocity_remote;
  return usd;
}

const UKFState UKFState::from_vector(const Matrix<double, 15, 1> vec) {
  return UKFState(
    vec.block(0, 0, 3, 1),
    vec.block(3, 0, 3, 1),
    vec.block(6, 0, 3, 1),
    vec.block(9, 0, 3, 1),
    vec.block(12, 0, 3, 1)
  );
}

ClingerUKF::ClingerUKF(double dt_, double trs_variance, double rot_variance, Matrix<double, 6, 6> pnp_R_, Matrix3d gyro_R_, double alpha, double beta, double kappa):
  dt(dt_), pnp_R(pnp_R_), gyro_R(gyro_R_) {
  // Calculate sigma point weightings, from van der Merwe formulation
  this->lambda = alpha * alpha * (n + kappa) - n;
  this->W0m = this->lambda / (this->n + this->lambda);
  this->W0c = this->W0m + 1 - alpha*alpha + beta;
  this->Wim = this->Wic = 1. / (2.*(this->n + this->lambda));
  // Store 2 seconds of history. Anything older than that is stale.
  n_history = 2. / dt;

  // Build the total Q matrix from one translation and a double rotation variance
  Matrix<double, 6, 6> posQ = discreteWhiteNoise3d2ndOrder(dt, trs_variance);
  Matrix<double, 6, 6> rotQ = discreteWhiteNoise3d2ndOrder(dt, rot_variance);
  Q.block(0,0,6,6) = posQ;
  Q.block(6,6,6,6) = rotQ;
  Q.block(12,12,3,3) = rotQ.block(3,3,3,3);
  this->P = this->Q;
}

std::vector<UKFState> ClingerUKF::sigmaPoints(UKFState mean, CovarMatrix covar) {
  // Use cholesky decomposition to compute the matrix square root
  CovarMatrix U(((this->n + this->lambda) * covar).llt().matrixU());
  int nsigma = 2*this->n + 1;
  std::vector<UKFState> sigmapts(nsigma);
  sigmapts[0] = mean;
  // Calculate sigma points spread around the mean
  // Pretty straightforward since most of the computation happens in operator overloads
  for (int k = 0; k < this->n; ++k) {
    UKFState sigma_d = UKFState::from_vector(U.block(0, k, 15, 1));
    sigmapts[k+1] = mean + sigma_d;
    sigmapts[k+this->n+1] = mean - sigma_d;
  }
  return sigmapts;
}

void ClingerUKF::predict(double predict_time) {
  std::vector<UKFState> sigma_points = this->sigmaPoints(this->state, this->P);
  int nsigma = sigma_points.size();
  std::vector<Quaterniond> sigma_quats;

  // Calculate the new mean by weighted sum of the sigma points
  std::vector<double> mean_weights(nsigma, this->Wim);
  mean_weights[0] = this->W0m;
  UKFState newMean;
  for (int i = 0; i < nsigma; ++i) {
    // Propagate the sigma points forward in time
    sigma_points[i] = sigma_points[i].integrate(dt);
    sigma_quats.push_back(sigma_points[i].orientation);
    // Quaternions don't get affected in this multiplication, but we replace those anyway later
    newMean = newMean + sigma_points[i] * mean_weights[i];
  }
  std::vector<Vector3d> unweighted_error_vectors;
  // Replace the new mean orientation with the quaternion mean
  // Also get the unweighted error vectors to build the new covariance matrix
  quaternionMean(sigma_quats, mean_weights, newMean.orientation, unweighted_error_vectors);
  CovarMatrix newP = CovarMatrix::Zero();
  for (int i = 0; i < nsigma; ++i) {
    // Build the new covariance matrix from the sigma points
    UKFState sigma = sigma_points[i];
    UKFState res = sigma - newMean;
    Matrix<double, 15, 1> residVec;
    residVec.block(0, 0, 3, 1) = res.position;
    residVec.block(3, 0, 3, 1) = res.velocity;
    residVec.block(6, 0, 3, 1) = unweighted_error_vectors[i];
    residVec.block(9, 0, 3, 1) = res.remoteRate;
    residVec.block(12, 0, 3, 1) = res.localRate;
    newP += (i == 0 ? this->W0c : this->Wic) * (residVec * residVec.transpose());
  }
  newP += this->Q;
  this->state = newMean;
  this->P = newP;
  recordStateHistory(predict_time, newMean, newP, sigma_points, unweighted_error_vectors);
}

void ClingerUKF::recordStateHistory(const double time, const UKFState state, const CovarMatrix P, const std::vector<UKFState> sigma_points, const std::vector<Vector3d> unweighted_error_vectors) {
  this->ukf_history.push_back(UKFHistoryElement(time, state, P, sigma_points, unweighted_error_vectors));
  if (this->ukf_history.size() > n_history) {
    this->ukf_history.pop_front();
  }
}
UKFHistoryElement ClingerUKF::getStateAtTime(const double time) {
  for (std::deque<UKFHistoryElement>::reverse_iterator el = ukf_history.rbegin(); el != ukf_history.rend(); ++el) {
    if (el->time <= time) {
      return *el;
    }
  }
  return UKFHistoryElement(-1, UKFState(), CovarMatrix::Zero(), std::vector<UKFState>(), std::vector<Vector3d>(31, Vector3d::Zero()));
}

void ClingerUKF::updatePnP(const double update_time, const double meas_time, const Vector3d trs, const Quaterniond rot) {
  UKFHistoryElement latest_state = this->ukf_history.back();
  UKFHistoryElement state_at_meas = getStateAtTime(meas_time);
  UKFState state = latest_state.state;
  std::vector<UKFState> sigma_points = latest_state.sigma_points;
  std::vector<Vector3d> unweighted_error_vectors = latest_state.unweighted_error_vectors;

  // We don't actually want to replay states, so just take the error at the time of the measurement and apply it to the current state
  // This doesn't properly propagate covariance but it's fast and easy and good enough
  UKFState delta_state = UKFState(trs, state_at_meas.state.velocity, rot, state_at_meas.state.remoteRate, state_at_meas.state.localRate) - state_at_meas.state;
  UKFState meas_state = state + delta_state;

  // Compute the innovation (sensor residual) as a vector in axis-angle format
  UKFState innovation = meas_state - state;
  Matrix<double, 6, 1> innovec;
  innovec.block(0,0,3,1) = innovation.position;
  AngleAxisd innov_aa(innovation.orientation);
  innovec.block(3,0,3,1) = innov_aa.axis() * innov_aa.angle();

  // Build up the measurement gain matrices
  int nsigma = latest_state.sigma_points.size();
  Matrix<double, 6, 6> Pzz = Matrix<double, 6, 6>::Zero();
  Matrix<double, 15, 6> Pxz = Matrix<double, 15, 6>::Zero();
  for (int i = 0; i < nsigma; ++i) {
    UKFState sigma = sigma_points[i];
    UKFState res = sigma - state;
    Matrix<double, 6, 1> meas_residVec;
    Matrix<double, 15, 1> residVec;
    residVec.block(0, 0, 3, 1) = res.position;
    residVec.block(3, 0, 3, 1) = res.velocity;
    residVec.block(6, 0, 3, 1) = unweighted_error_vectors[i];
    residVec.block(9, 0, 3, 1) = res.remoteRate;
    residVec.block(12, 0, 3, 1) = res.localRate;
    meas_residVec.block(0, 0, 3, 1) = res.position;
    meas_residVec.block(3, 0, 3, 1) = unweighted_error_vectors[i];
    Pxz += (i == 0 ? this->W0c : this->Wic) * (residVec * meas_residVec.transpose());
    Pzz += (i == 0 ? this->W0c : this->Wic) * (meas_residVec * meas_residVec.transpose());
  }
  Pzz += pnp_R;
  // Calculate the kalman gain and apply it to the innovation
  Matrix<double, 15, 6> K = Pxz * Pzz.inverse();
  CovarMatrix new_P = P - K * Pzz * K.transpose();
  // Update the state
  UKFState newState = state + UKFState::from_vector(K * innovec);
  this->state = newState;
  this->P = new_P;
  recordStateHistory(update_time, newState, new_P, sigma_points, unweighted_error_vectors);
}

void ClingerUKF::updateLocalGyro(const double update_time, const double meas_time, const Vector3d rate) {
  UKFHistoryElement latest_state = this->ukf_history.back();
  UKFHistoryElement state_at_meas = getStateAtTime(meas_time);
  UKFState state = latest_state.state;
  std::vector<UKFState> sigma_points = latest_state.sigma_points;
  std::vector<Vector3d> unweighted_error_vectors = latest_state.unweighted_error_vectors;

  // We don't actually want to replay states, so just take the error at the time of the measurement and apply it to the current state
  // This doesn't properly propagate covariance but it's fast and easy and good enough
  UKFState delta_state = UKFState(state_at_meas.state.position, state_at_meas.state.velocity, state_at_meas.state.orientation, state_at_meas.state.remoteRate, rate) - state_at_meas.state;
  UKFState meas_state = state + delta_state;

  // Compute the innovation
  UKFState innovation = meas_state - state;
  Matrix<double, 3, 1> innovec = innovation.localRate;

  int nsigma = latest_state.sigma_points.size();
  // Build up the measurement gain matrices
  Matrix<double, 3, 3> Pzz = Matrix<double, 3, 3>::Zero();
  Matrix<double, 15, 3> Pxz = Matrix<double, 15, 3>::Zero();
  for (int i = 0; i < nsigma; ++i) {
    UKFState sigma = sigma_points[i];
    UKFState res = sigma - state;
    Matrix<double, 3, 1> meas_residVec;
    Matrix<double, 15, 1> residVec;
    residVec.block(0, 0, 3, 1) = res.position;
    residVec.block(3, 0, 3, 1) = res.velocity;
    residVec.block(6, 0, 3, 1) = unweighted_error_vectors[i];
    residVec.block(9, 0, 3, 1) = res.remoteRate;
    residVec.block(12, 0, 3, 1) = res.localRate;
    meas_residVec.block(0, 0, 3, 1) = res.localRate;
    Pxz += (i == 0 ? this->W0c : this->Wic) * (residVec * meas_residVec.transpose());
    Pzz += (i == 0 ? this->W0c : this->Wic) * (meas_residVec * meas_residVec.transpose());
  }
  Pzz += gyro_R;
  // Compute the kalman gain and apply it to the innovation to update the state estimate
  Matrix<double, 15, 3> K = Pxz * Pzz.inverse();
  CovarMatrix new_P = P - K * Pzz * K.transpose();
  UKFState newState = state + UKFState::from_vector(K * innovec);
  this->state = newState;
  this->P = new_P;
  recordStateHistory(update_time, newState, new_P, sigma_points, unweighted_error_vectors);
}

void ClingerUKF::updateRemoteGyro(const double update_time, const double meas_time, const Vector3d rate) {
  UKFHistoryElement latest_state = this->ukf_history.back();
  UKFHistoryElement state_at_meas = getStateAtTime(meas_time);
  UKFState state = latest_state.state;
  std::vector<UKFState> sigma_points = latest_state.sigma_points;
  std::vector<Vector3d> unweighted_error_vectors = latest_state.unweighted_error_vectors;

  // We don't actually want to replay states, so just take the error at the time of the measurement and apply it to the current state
  // This doesn't properly propagate covariance but it's fast and easy and good enough
  UKFState delta_state = UKFState(state_at_meas.state.position, state_at_meas.state.velocity, state_at_meas.state.orientation, rate,state_at_meas.state.localRate) - state_at_meas.state;
  UKFState meas_state = state + delta_state;

  // Compute the innovation
  UKFState innovation = meas_state - state;
  Matrix<double, 3, 1> innovec = innovation.remoteRate;

  int nsigma = latest_state.sigma_points.size();
  // Build up the sensor gain matrices
  Matrix<double, 3, 3> Pzz = Matrix<double, 3, 3>::Zero();
  Matrix<double, 15, 3> Pxz = Matrix<double, 15, 3>::Zero();
  for (int i = 0; i < nsigma; ++i) {
    UKFState sigma = sigma_points[i];
    UKFState res = sigma - state;
    Matrix<double, 3, 1> meas_residVec;
    Matrix<double, 15, 1> residVec;
    residVec.block(0, 0, 3, 1) = res.position;
    residVec.block(3, 0, 3, 1) = res.velocity;
    residVec.block(6, 0, 3, 1) = unweighted_error_vectors[i];
    residVec.block(9, 0, 3, 1) = res.remoteRate;
    residVec.block(12, 0, 3, 1) = res.localRate;
    meas_residVec.block(0, 0, 3, 1) = res.remoteRate;
    Pxz += (i == 0 ? this->W0c : this->Wic) * (residVec * meas_residVec.transpose());
    Pzz += (i == 0 ? this->W0c : this->Wic) * (meas_residVec * meas_residVec.transpose());
  }
  Pzz += gyro_R;
  // Compute the kalman gain and apply it to the innovation to update the state estimate
  Matrix<double, 15, 3> K = Pxz * Pzz.inverse();
  CovarMatrix new_P = P - K * Pzz * K.transpose();
  UKFState newState = state + UKFState::from_vector(K * innovec);
  this->state = newState;
  this->P = new_P;
  recordStateHistory(update_time, newState, new_P, sigma_points, unweighted_error_vectors);
}

void testUKF() {
  Quaterniond Rbegin = zyx(radians(-30), radians(10), radians(4));
  Quaterniond Rend = zyx(radians(30), radians(10), radians(-4));
  Vector3d posbegin(-10, 0, 3);
  Vector3d posend(10, 0, -3);
  double t0 = 0;
  double tend = 10;
  double dt = 0.1;

  auto pnp_R = 0.2 * Matrix<double, 6, 6>::Identity();
  auto gyro_R = 0.1 * Matrix<double, 3, 3>::Identity();
  auto remote_rate = Vector3d::Zero();
  auto local_rate = Vector3d(radians(-0.8), 0, radians(6));
  ClingerUKF ukf(dt, 0.1, 0.1, pnp_R, gyro_R, 0.1, 2, -12);
  ukf.state.position = posbegin;
  ukf.state.orientation = Rbegin;
  
  for (double t = t0; t < tend; t += dt) {
    double param = (t - t0) / (tend - t0);
    Quaterniond rot = Rbegin.slerp(param, Rend);
    Vector3d pos = param * (posend - posbegin) + posbegin;
    auto cycle_begin = system_clock::now();
    ukf.predict(t);
    auto predict_time = system_clock::now();
    ukf.updatePnP(t, t, pos, rot);
    ukf.updateRemoteGyro(t, t, remote_rate);
    ukf.updateLocalGyro(t, t, local_rate);
    auto update_time = system_clock::now();
    Vector3d pos_error = ukf.state.position - pos;
    Quaterniond ori_error = rot.inverse() * ukf.state.orientation;
    auto predict_ms = std::chrono::duration_cast<std::chrono::microseconds>(predict_time - cycle_begin).count();
    auto update_ms = std::chrono::duration_cast<std::chrono::microseconds>(update_time - predict_time).count();
    std::cout << "Predict: " << predict_ms << " us, update " << update_ms << " us" << std::endl;
    std::cout << "Position: " << ukf.state.position.transpose() << " Velocity: " << ukf.state.velocity.transpose() << std::endl;
    std::cout << "Remote rate: " << ukf.state.remoteRate.transpose() << " Local Rate: " << ukf.state.localRate.transpose() << std::endl;
    std::cout << "Position Error: " << pos_error.transpose() << " Orientation Error: " << degrees(rot.angularDistance(ukf.state.orientation)) << " deg" << std::endl << std::endl;
  }
}

int main(int argc, char *argv[]) {
  testUKF();
}