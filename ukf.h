#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include <chrono>
#include <deque>

using namespace Eigen;
using namespace std::chrono;

/**
 * Implement the iterative weighted quaternion geodesic mean, from Kraft's paper
 * 
 * @param quats the list of quaternions
 * @param weights the corresponding weights (need not sum to 1)
 * @param res the resulting weighted mean quaternion
 * @param errorVecs the resulting last unweighted error vectors
 */
void quaternionMean(std::vector<Quaterniond> quats, std::vector<double> weights, Quaterniond& res, std::vector<Vector3d>& errorVecs);

/**
 * 
 * @param dt the discretization step in seconds
 * @param variance the variance of the first order of the process
 * @returns the discretized Q matrix for a 3-dimensional 2nd-order (position/velocity etc) process
 */
Matrix<double, 6, 6> discreteWhiteNoise3d2ndOrder(const double dt, const double variance);

/**
 * Create a ZYX rotation from euler angles
 * 
 * @param z z/yaw (radians)
 * @param y y/pitch (radians)
 * @param x x/roll (radians)
 * @return the total rotation
 */
Quaterniond zyx(const double z, const double y, const double x);

/**
 * @param degrees degree angle
 * @returns `degrees` converted to radians
 */
constexpr double radians(double degrees) {
  return degrees * M_PI / 180.0;
}

/**
 * @param radians radian angle
 * @returns `radians` converted to degrees
 */
constexpr double degrees(double radians) {
  return radians * 180. / M_PI;
}

std::vector<double> eigen_to_list(const MatrixXd mat) {
  return std::vector<double>(mat.data(), mat.data() + mat.size());
}

/**
 * @returns the number of seconds since the epoch with at least millisecond precision.
 */
double timestamp() {
  milliseconds ms = duration_cast< milliseconds >(
      system_clock::now().time_since_epoch()
  );
  return ((double) ms.count()) / 1000.0;
}

/**
 * Templated Runge-Kutta fourth-order integration function.
 * 
 * For any integration functions, the following operators must be defined:
 * T = T + V
 * V = V * scalar
 * V = V + V
 * 
 * If these are defined this function will /just work/. For basic usage, you can template it with Eigen::VectorN and Eigen::VectorN.
 * You only need to create custom types if addition is wonky (like quaternions are)
 * 
 * @tparam T the type of the left hand side, representing a point in space.
 * @tparam V the type of the right hand side, representing a difference between two points. The generalized derivative type of T.
 * @param x0 the beginning point
 * @param x_dot the function used to calculate the derivative at (x, t)
 * @param t0 the begin time
 * @param t_step delta time
 * @returns x_dot integrated along t_step starting at x0
 */
template<typename T, typename V> T integrateRK4(T x0, std::function<V(T, double)> x_dot, double t0, double t_step) {
  V k1 = x_dot(x0, t0);
  double inner_step = t_step / 2;
  V k2 = x_dot(x0 + k1 * inner_step, t0 + inner_step);
  V k3 = x_dot(x0 + k2 * inner_step, t0 + inner_step);
  V k4 = x_dot(x0 + k3 * t_step, t0 + t_step);
  return x0 + (k1 + k2*2 + k3*2 + k4) * (t_step / 6.0);
}
/**
 * Templated zero-order hold (forward Euler) integration function.
 * 
 * @tparam T the type of the left hand side, representing a point in space.
 * @tparam V the type of the right hand side, representing a difference between two points. The generalized derivative type of T.
 * @param x0 the beginning point
 * @param x_dot the function used to calculate the derivative at (x, t)
 * @param t0 the begin time
 * @param t_step delta time
 * @returns x_dot integrated along t_step starting at x0
 */
template<typename T, typename V> T integrateZOH(T x0, std::function<V(T, double)> x_dot, double t0, double t_step) {
  V k1 = x_dot(x0, t0);
  return x0 + (k1 * t_step);
}

/**
 * Represents a difference or derivative of a UKFState
 */
class UKFStateDerivative {
  public:
  Vector3d velocity_remote;
  Vector3d body_rate_local;
  Vector3d acceleration_remote;
  Vector3d frame_rate;

  UKFStateDerivative operator*(double d);
  UKFStateDerivative operator+(UKFStateDerivative other);
};

/**
 * Represents the state of a CLINGERs unit
 */
class UKFState {
  public:
  UKFState(): position(Vector3d::Zero()), velocity(Vector3d::Zero()), orientation(Quaterniond::Identity()), remoteRate(Vector3d::Zero()), localRate(Vector3d::Zero()) {};
  UKFState(const Vector3d position_, const Vector3d velocity_, const Quaterniond orientation_, const Vector3d remoteRate_, const Vector3d localRate_):
    position(position_), velocity(velocity_), orientation(orientation_), remoteRate(remoteRate_), localRate(localRate_) {};
  UKFState(const Vector3d position_, const Vector3d velocity_, const Vector3d orientation_, const Vector3d remoteRate_, const Vector3d localRate_):
    position(position_), velocity(velocity_), orientation(AngleAxisd(orientation_.norm(), orientation_.normalized())), remoteRate(remoteRate_), localRate(localRate_) {};
  static const UKFState from_vector(const Matrix<double, 15, 1> vec);

  const UKFState operator+(const UKFState other);
  const UKFState operator*(const double d);
  const UKFState operator-(const UKFState other);
  const UKFState operator-();
  const UKFState operator+(const UKFStateDerivative other);
  const UKFStateDerivative derivative();
  const UKFState integrate(const double dt);

  Vector3d position;
  Vector3d velocity;
  Quaterniond orientation;
  Vector3d localRate;
  Vector3d remoteRate;
};

/**
 * Convenience type to represent a 15x15 matrix, commonly used to represent the covariance of the UKF
 */
typedef Matrix<double, 15, 15> CovarMatrix;

/**
 * Dataclass to store a state in the past
 */
class UKFHistoryElement {
  public:
  UKFHistoryElement(double time_, UKFState state_, CovarMatrix P_, std::vector<UKFState> sigma_points_, std::vector<Vector3d> unweighted_error_vectors_):
    time(time_), state(state_), P(P_), sigma_points(sigma_points_), unweighted_error_vectors(unweighted_error_vectors_) {};
  double time;
  UKFState state;
  CovarMatrix P;
  std::vector<UKFState> sigma_points;
  std::vector<Vector3d> unweighted_error_vectors;
};

/**
 * Implements an unscented Kalman filter to estimate the state of the CLINGERs unit.
 * See https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf for the quaternion components.
 */
class ClingerUKF {
  public:
  /**
   * @param dt the predict dt (seconds)
   * @param trs_variance the expected white noise of the translational process (m^2)
   * @param rot_variance the expected white noise of the rotational process (rad^2)
   * @param pnp_R the R matrix for PnP measurements. Units [m m m rad rad rad]^2
   * @param gyro_R the R matrix for gyro measurements. Units [rad/s rad/s rad/s]^2
   * @param alpha the alpha parameter for the sigma point calculation. (0, 1].
   * @param beta the beta parameter for the sigma point calculation. Recommend 2.
   * @param kappa the kappa parameter for the sigma point calculation. Recommend 3-15 = -12
   */
  ClingerUKF(const double dt, const double trs_variance, const double rot_variance, const Matrix<double, 6, 6> pnp_R, const Matrix3d gyro_R, const double alpha, const double beta, const double kappa);
  /**
   * Run the UKF predict step.
   * @param predict_time the time of prediction (use `timestamp()`)
   */
  void predict(const double predict_time);
  /**
   * Run the UKF update step for the PnP sensor.
   * @param update_time the time of the update (use `timestamp()`)
   * @param meas_time the time of the measurement (image frame time)
   * @param trs the translation measurement
   * @param rot the rotation measurement
   */
  void updatePnP(const double update_time, const double meas_time, const Vector3d trs, const Quaterniond rot);
  /**
   * Run the UKF update step for the local gyro sensor (on this CLINGER)
   * @param update_time the time of the update (use `timestamp()`)
   * @param meas_time the time of the measurement
   * @param rate the gyro rate
   */
  void updateLocalGyro(const double update_time, const double meas_time, const Vector3d rate);
  /**
   * Run the UKF update step for the remote gyro sensor (on the other CLINGER)
   * @param update_time the time of the update (use `timestamp()`)
   * @param meas_time the time of the measurement
   * @param rate the gyro rate
   */
  void updateRemoteGyro(const double update_time, const double meas_time, const Vector3d rate);
  UKFHistoryElement getStateAtTime(const double time);
  UKFState state;

  private:
  int n_history;
  void recordStateHistory(const double time, const UKFState state, const CovarMatrix P, const std::vector<UKFState> sigma_points, const std::vector<Vector3d> unweighted_error_vectors);
  CovarMatrix Q;
  Matrix<double, 6, 6> pnp_R;
  Matrix<double, 3, 3> gyro_R;
  CovarMatrix P;
  double lambda;
  double dt;
  double W0m, W0c, Wic, Wim;
  int n = 15;
  std::deque<UKFHistoryElement> ukf_history;
  /**
   * Calculate the sigma points, using the van der Merwe formulation
   * @param mean the current mean state estimate
   * @param covar the current covariance (P) matrix
   * @return a vector of sigma points
   */
  std::vector<UKFState> sigmaPoints(UKFState mean, CovarMatrix covar);
};