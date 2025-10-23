from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# ✅ Set your project URL here
LOGIN_URL = "http://127.0.0.1:5000"  # update if needed

# ✅ Test credentials
USERNAME = "Meena"
PASSWORD = "meena123"

# Launch Chrome
driver = webdriver.Chrome()
driver.maximize_window()

try:
    print("Opening SmartCart Login Page...")
    driver.get(LOGIN_URL)
    time.sleep(2)

    # Fill in the name and password fields
    name_input = driver.find_element(By.ID, "name")
    password_input = driver.find_element(By.ID, "password")

    name_input.send_keys(USERNAME)
    password_input.send_keys(PASSWORD)

    # Click the Login button
    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    login_button.click()
    print("Attempting to log in...")

    # Wait for redirect to dashboard (index_page)
    time.sleep(3)

    # Check if redirected successfully
    if "index_page" in driver.current_url:
        print("✅ Login Test Passed: Redirected to index_page.")
    else:
        print("❌ Login Test Failed: Not redirected correctly.")
        print("Current URL:", driver.current_url)

except Exception as e:
    print("❌ Test failed due to error:", str(e))

finally:
    # Keep browser open for observation
    input("\nPress Enter to close the browser...")
    driver.quit()
