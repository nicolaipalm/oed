import numpy as np

#######
from ..parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)

# aging model


class Aging_Model_Cal:
    def __init__(
            self,
            SoCref: float = 0.5,
            Tref: float = 318.15,
            EOL_C: float = 0.9561,
            t_end: float = 287,

            #t: float,
    ):
        """
        Parameters
        ----------
        SoCref :
        Tref :
        EOL_C :
        t_end :
        t :
        """

        # define reference values
        self.SoCref = SoCref  # in p.u.
        self.Tref = Tref  # in K
        self.EOL_C = EOL_C  # in p.u.
        self.t_end = t_end  # in days

        #self.t = t

    # Calendar Aging model

    def x_ref_cal(self, param: float) -> float:
        return (1 - self.EOL_C) / ((self.t_end * (24 * 3600)) ** param)

    def d_SoC_cal(self, SoC: float, param: np.ndarray) -> float:
        return (SoC / self.SoCref) ** (1 / param[0])

    def d_T_cal(self, T: float, param: np.ndarray) -> float:
        return np.exp(-param[0] * ((1 / T) - (1 / self.Tref)))

    def Calendar_Aging(self, theta: np.ndarray, x: np.ndarray) -> float:
        """
        theta = [theta0, theta1, theta2] .parameters of aging model
        x = [SoC (0..1), T (in K), t (0, 7, 14,.. in days)] .......independent quantities

        :return: Q_loss_cal
        """

        Q_loss_cal = (
                self.x_ref_cal(theta[2])
                * self.d_SoC_cal(x[0], np.array([theta[0]]))
                * self.d_T_cal(x[1], np.array([theta[1]]))
                * (x[2] * (24 * 3600)) ** (theta[2])
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
                    np.log(self.t_end * (24 * 3600)) - np.log(self.x[2] * (24 * 3600))
            )

        else:
            pass


#############################


class AgingModelCal(ParametricFunction):
    def __init__(self):
        self._aging_model = Aging_Model_Cal()

    def __call__(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self._aging_model.Calendar_Aging(theta=theta, x=x)

    def partial_derivative(
            self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        return self._aging_model.partial_derivative_Calendar_Aging(theta=theta, x=x, index=parameter_index, )

    def second_partial_derivative(
            self,
            theta: np.ndarray,
            x: np.ndarray,
            parameter1_index: int,
            parameter2_index: int,
    ) -> np.ndarray:
        raise NotImplementedError
