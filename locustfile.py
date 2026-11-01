
from locust import HttpUser, task, between


class WalmartAPIUser(HttpUser):
    wait_time = between(1, 2)

    sample = {
        "Store": 1,
        "Holiday_Flag": 0,
        "Temperature": 42.31,
        "Fuel_Price": 2.572,
        "CPI": 211.096358,
        "Unemployment": 8.106,
        "Year": 2010,
        "Month": 2,
        "Week": 5,
        "Quarter": 1,
    }

    @task(3)
    def predict_sales(self):
        with self.client.post(
            "/predict",
            json=self.sample,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "prediction" in result:
                    response.success()
                else:
                    response.failure("Prediction missing in response")
            else:
                response.failure(
                    f"HTTP {response.status_code}: {response.text}"
                )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
