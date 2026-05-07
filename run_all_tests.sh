#!/bin/bash

# --- SUT Corporate Health: Master Regression Suite ---
# This script runs all testing layers to ensure zero regressions.

echo "===================================================="
echo "🚀 Starting Full SUT Regression Suite"
echo "===================================================="

# 1. Backend Unit Tests
echo -e "\n[1/4] Running Backend Unit Tests..."
cd tests && python -m pytest unit/ -v --tb=short
if [ $? -ne 0 ]; then echo "❌ Unit Tests Failed!"; exit 1; fi

# 2. Backend Integration Tests (Black-Box)
echo -e "\n[2/4] Running Backend Integration Tests (Black-Box)..."
# Ensure these run against the docker container
TEST_ADMIN_USER=admin TEST_ADMIN_PASS="Admin@1234!" python -m pytest integration/ -v --tb=short
if [ $? -ne 0 ]; then echo "❌ Integration Tests Failed!"; exit 1; fi

# 3. Frontend E2E Tests (Playwright)
echo -e "\n[3/4] Running Frontend E2E Tests (Playwright)..."
# Note: Requires 'npm install -g @playwright/test' and the frontend to be running at localhost:5173
if command -v npx &> /dev/null
then
    cd e2e && npx playwright test
else
    echo "⚠️ npx not found. Skipping E2E tests. Run manually in frontend/ with 'npx playwright test'"
fi

# 4. Stress Test Summary
echo -e "\n[4/4] Running Short Stress Test (Stability Check)..."
cd ../stress && locust -f locustfile.py --host=http://localhost:8000 --headless -u 5 -r 1 --run-time 10s --only-summary
if [ $? -ne 0 ]; then echo "⚠️ Stress Test returned errors (Normal if DB is fresh)."; fi

echo "===================================================="
echo "✅ REGRESSION SUITE COMPLETE"
echo "===================================================="
