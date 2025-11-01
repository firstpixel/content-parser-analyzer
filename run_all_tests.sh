#!/bin/bash

echo "=========================================="
echo "Running Complete Test Suite"
echo "=========================================="
echo ""

echo "1. Original Tests (Backward Compatibility)"
echo "-------------------------------------------"
python test_content_parser.py > /tmp/test1.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PASSED: Original tests"
else
    echo "❌ FAILED: Original tests"
    cat /tmp/test1.log
    exit 1
fi
echo ""

echo "2. Advanced NLP Analyzer Unit Tests"
echo "-------------------------------------------"
python test_advanced_nlp_analyzer.py > /tmp/test2.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PASSED: Advanced NLP unit tests"
else
    echo "❌ FAILED: Advanced NLP unit tests"
    cat /tmp/test2.log
    exit 1
fi
echo ""

echo "3. Integration Tests"
echo "-------------------------------------------"
python test_integration_advanced.py > /tmp/test3.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ PASSED: Integration tests"
else
    echo "❌ FAILED: Integration tests"
    cat /tmp/test3.log
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ ALL TESTS PASSED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Test Summary:"
echo "  - Original tests: PASSED"
echo "  - Advanced NLP unit tests: PASSED"
echo "  - Integration tests: PASSED"
echo ""
echo "Total: 3/3 test suites passing"
