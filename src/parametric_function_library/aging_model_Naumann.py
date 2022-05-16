import numpy as np

#######
from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)

# aging model


class Aging_Model_Nau:
    def __init__(
        self,
        Tref: float = 297.15,
        SoCref: float = 0.5,
        C_rest_cal: float = 0.1,
        t_end_ref: float = 520,
        DoDref: float = 0.8,
        crateref: float = 1,
        C_rest_cyc: float = 0.1,
        FEC_end_ref: float = 4500,
    ):

        # define reference values
        self.Tref = Tref  # in K
        self.SoCref = SoCref  # in p.u.
        self.C_rest_cal = C_rest_cal  # in p.u.
        self.t_end_ref = t_end_ref  # in days

        self.DoDref = DoDref  # in p.u.
        self.crateref = crateref  # in p.u.
        self.C_rest_cyc = C_rest_cyc  # in p.u.
        self.FEC_end_ref = FEC_end_ref

    # Calendar Aging model

    def x_ref_cal(self, param: float) -> float:
        return np.sqrt(self.C_rest_cal / ((self.t_end_ref * (24 * 3600)) ** param))

    def d_SoC_cal(self, SoC: float, param: np.ndarray) -> float:
        return self.x_ref_cal(param[1]) * ((SoC / self.SoCref) ** (1 / param[0]))

    def d_T_cal(self, T: float, param: np.ndarray) -> float:
        return self.x_ref_cal(param[1]) * np.exp(
            -param[0] * ((1 / T) - (1 / self.Tref))
        )

    def Calendar_Aging(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        theta = [theta0, theta1, theta2] .parameters of aging model
        x = [SoC (0..1), T (in K), t (0, 7, 14,.. in days)] .......independent quantities

        :return: Q_loss_cal
        """

        Q_loss_cal = (
            self.d_SoC_cal(x[0], np.array([theta[0], theta[2]]))
            * self.d_T_cal(x[1], np.array([theta[1], theta[2]]))
            * (x[-1] * (24 * 3600)) ** (theta[2])
        )

        return Q_loss_cal

    def partial_derivative_Calendar_Aging(
        self, theta: np.ndarray, x: np.ndarray, index: int
    ) -> np.ndarray:
        """
        theta = [theta0, theta1, theta2] .parameters of aging model
        x = [SoC (0..1), T (in K), t (0, 7, 14,.. in days)] .......independent quantities
        index = 0..2 ....parameter selection for partial derivative

        :return: partial derivative of Calendar Aging capacity model
        """

        if index == 0:
            return (
                -self.Calendar_Aging(theta, x)
                * np.log(x[0] / self.SoCref)
                / theta[0] ** 2
            )

        elif index == 1:
            return -self.Calendar_Aging(theta, x) * ((1 / x[1]) - (1 / self.Tref))

        elif index == 2:
            return -self.Calendar_Aging(theta, x) * (
                np.log(self.t_end_ref * (24 * 3600)) - np.log(x[-1] * (24 * 3600))
            )

        else:
            pass


#############################


class AgingModelNaumann(ParametricFunction):
    def __call__(self, theta: np.ndarray, x: np.ndarray) -> float:
        if np.sum(Aging_Model_Nau().Calendar_Aging(theta=theta, x=x)) < 0:
            print(np.sum(Aging_Model_Nau().Calendar_Aging(theta=theta, x=x)), x, theta)

        if np.sum(Aging_Model_Nau().Calendar_Aging(theta=theta, x=x)) > 10000:
            print(np.sum(Aging_Model_Nau().Calendar_Aging(theta=theta, x=x)), x, theta)

        return np.sum(Aging_Model_Nau().Calendar_Aging(theta=theta, x=x))

    def partial_derivative(
        self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> float:
        return np.sum(
            Aging_Model_Nau().partial_derivative_Calendar_Aging(
                theta=theta, x=x, index=parameter_index
            )
        )

    def second_partial_derivative(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        parameter1_index: int,
        parameter2_index: int,
    ) -> float:
        raise NotImplementedError
