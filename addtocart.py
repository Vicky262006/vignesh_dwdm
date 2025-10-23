from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

# ---------------- CONFIG ----------------
SAUCE_URL = "https://www.saucedemo.com/"
USERNAME = "standard_user"
PASSWORD = "secret_sauce"

# ---------------- CHROME OPTIONS ----------------
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-extensions")

driver = webdriver.Chrome(options=chrome_options)

try:
    # ---------------- LOGIN ----------------
    driver.get(SAUCE_URL)
    time.sleep(2)  # wait for page to settle
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "user-name")))
    
    driver.find_element(By.ID, "user-name").send_keys(USERNAME)
    time.sleep(1)
    driver.find_element(By.ID, "password").send_keys(PASSWORD)
    time.sleep(1)
    driver.find_element(By.ID, "login-button").click()
    
    # Wait for inventory page
    WebDriverWait(driver, 10).until(EC.url_contains("inventory.html"))
    print("✅ Login Successful")
    time.sleep(2)

    # ---------------- ADD ITEMS TO CART ----------------
    # Add first two products to cart
    add_buttons = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "btn_inventory"))
    )
    add_buttons[0].click()
    print("✅ Added 1st item")
    time.sleep(2)
    add_buttons[1].click()
    print("✅ Added 2nd item")
    time.sleep(2)

    # ---------------- OPEN CART ----------------
    driver.find_element(By.ID, "shopping_cart_container").click()
    WebDriverWait(driver, 10).until(EC.url_contains("cart.html"))
    print("✅ Cart opened with items")
    time.sleep(2)

    # ---------------- LOGOUT ----------------
    # Open menu
    driver.find_element(By.ID, "react-burger-menu-btn").click()
    time.sleep(2)  # wait for menu animation
    driver.find_element(By.ID, "logout_sidebar_link").click()
    
    # Wait for redirect back to login page
    WebDriverWait(driver, 10).until(EC.url_contains("saucedemo.com"))
    print("✅ Logout Successful")
    time.sleep(2)

except Exception as e:
    print("❌ Test Error:", str(e))

finally:
    driver.quit()
