/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(self["webpackChunk_N_E"] = self["webpackChunk_N_E"] || []).push([["pages/index"],{

/***/ "./node_modules/next/dist/build/webpack/loaders/next-client-pages-loader.js?absolutePagePath=C%3A%5CUsers%5Cdevli%5CDocuments%5CDevelopment%5Cmhacks%5Cwebsite%5Cclient%5Cpages%5Cindex.tsx&page=%2F!":
/*!************************************************************************************************************************************************************************************************************!*\
  !*** ./node_modules/next/dist/build/webpack/loaders/next-client-pages-loader.js?absolutePagePath=C%3A%5CUsers%5Cdevli%5CDocuments%5CDevelopment%5Cmhacks%5Cwebsite%5Cclient%5Cpages%5Cindex.tsx&page=%2F! ***!
  \************************************************************************************************************************************************************************************************************/
/***/ (function(module, __unused_webpack_exports, __webpack_require__) {

eval(__webpack_require__.ts("\n    (window.__NEXT_P = window.__NEXT_P || []).push([\n      \"/\",\n      function () {\n        return __webpack_require__(/*! ./pages/index.tsx */ \"./pages/index.tsx\");\n      }\n    ]);\n    if(true) {\n      module.hot.dispose(function () {\n        window.__NEXT_P.push([\"/\"])\n      });\n    }\n  //# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9ub2RlX21vZHVsZXMvbmV4dC9kaXN0L2J1aWxkL3dlYnBhY2svbG9hZGVycy9uZXh0LWNsaWVudC1wYWdlcy1sb2FkZXIuanM/YWJzb2x1dGVQYWdlUGF0aD1DJTNBJTVDVXNlcnMlNUNkZXZsaSU1Q0RvY3VtZW50cyU1Q0RldmVsb3BtZW50JTVDbWhhY2tzJTVDd2Vic2l0ZSU1Q2NsaWVudCU1Q3BhZ2VzJTVDaW5kZXgudHN4JnBhZ2U9JTJGISIsIm1hcHBpbmdzIjoiO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsZUFBZSxtQkFBTyxDQUFDLDRDQUFtQjtBQUMxQztBQUNBO0FBQ0EsT0FBTyxJQUFVO0FBQ2pCLE1BQU0sVUFBVTtBQUNoQjtBQUNBLE9BQU87QUFDUDtBQUNBIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vX05fRS8/NzY3NCJdLCJzb3VyY2VzQ29udGVudCI6WyJcbiAgICAod2luZG93Ll9fTkVYVF9QID0gd2luZG93Ll9fTkVYVF9QIHx8IFtdKS5wdXNoKFtcbiAgICAgIFwiL1wiLFxuICAgICAgZnVuY3Rpb24gKCkge1xuICAgICAgICByZXR1cm4gcmVxdWlyZShcIi4vcGFnZXMvaW5kZXgudHN4XCIpO1xuICAgICAgfVxuICAgIF0pO1xuICAgIGlmKG1vZHVsZS5ob3QpIHtcbiAgICAgIG1vZHVsZS5ob3QuZGlzcG9zZShmdW5jdGlvbiAoKSB7XG4gICAgICAgIHdpbmRvdy5fX05FWFRfUC5wdXNoKFtcIi9cIl0pXG4gICAgICB9KTtcbiAgICB9XG4gICJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./node_modules/next/dist/build/webpack/loaders/next-client-pages-loader.js?absolutePagePath=C%3A%5CUsers%5Cdevli%5CDocuments%5CDevelopment%5Cmhacks%5Cwebsite%5Cclient%5Cpages%5Cindex.tsx&page=%2F!\n"));

/***/ }),

/***/ "./pages/index.tsx":
/*!*************************!*\
  !*** ./pages/index.tsx ***!
  \*************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"./node_modules/react/jsx-dev-runtime.js\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ \"./node_modules/react/index.js\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);\n\nvar _s = $RefreshSig$();\n\nconst IndexPage = ()=>{\n    _s();\n    const [confidence, setConfidence] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)(0.0);\n    const [label, setLabel] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)(\"False\");\n    const [recording, setRecording] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)(false);\n    (0,react__WEBPACK_IMPORTED_MODULE_1__.useEffect)(()=>{\n        const eventSource = new EventSource(\"http://localhost:8080/audio_stream\");\n        eventSource.onmessage = (event)=>{\n            const [newConfidence, newLabel] = event.data.split(\",\");\n            setConfidence(parseFloat(newConfidence));\n            setLabel(newLabel);\n        };\n        return ()=>{\n            eventSource.close();\n        };\n    }, []);\n    const toggleRecording = async ()=>{\n        try {\n            if (recording) {\n                // If already recording, stop recording\n                setRecording(false);\n            } else {\n                // If not recording, start recording\n                const response = await fetch(\"http://localhost:8080/start_recording\", {\n                    method: \"POST\"\n                });\n                const data = await response.json();\n                console.log(data);\n                setRecording(true);\n            }\n        } catch (error) {\n            console.error(\"Error toggling recording:\", error);\n        }\n    };\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n        children: [\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"h1\", {\n                children: \"Audio Classification Demo\"\n            }, void 0, false, {\n                fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                lineNumber: 41,\n                columnNumber: 7\n            }, undefined),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                children: [\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"p\", {\n                        children: [\n                            \"Confidence: \",\n                            confidence.toFixed(2)\n                        ]\n                    }, void 0, true, {\n                        fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                        lineNumber: 43,\n                        columnNumber: 9\n                    }, undefined),\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"p\", {\n                        children: [\n                            \"Danger: \",\n                            label\n                        ]\n                    }, void 0, true, {\n                        fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                        lineNumber: 44,\n                        columnNumber: 9\n                    }, undefined),\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"progress\", {\n                        max: 1,\n                        value: confidence\n                    }, void 0, false, {\n                        fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                        lineNumber: 45,\n                        columnNumber: 9\n                    }, undefined)\n                ]\n            }, void 0, true, {\n                fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                lineNumber: 42,\n                columnNumber: 7\n            }, undefined),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"button\", {\n                onClick: toggleRecording,\n                children: recording ? \"Stop Recording\" : \"Start Recording\"\n            }, void 0, false, {\n                fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n                lineNumber: 47,\n                columnNumber: 7\n            }, undefined)\n        ]\n    }, void 0, true, {\n        fileName: \"C:\\\\Users\\\\devli\\\\Documents\\\\Development\\\\mhacks\\\\website\\\\client\\\\pages\\\\index.tsx\",\n        lineNumber: 40,\n        columnNumber: 5\n    }, undefined);\n};\n_s(IndexPage, \"CapX7F466fWDvV4Xn/HNIyF+ZfU=\");\n_c = IndexPage;\n/* harmony default export */ __webpack_exports__[\"default\"] = (IndexPage);\nvar _c;\n$RefreshReg$(_c, \"IndexPage\");\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9wYWdlcy9pbmRleC50c3giLCJtYXBwaW5ncyI6Ijs7Ozs7OztBQUFtRDtBQUVuRCxNQUFNRyxZQUFzQjs7SUFDMUIsTUFBTSxDQUFDQyxZQUFZQyxjQUFjLEdBQUdILCtDQUFRQSxDQUFTO0lBQ3JELE1BQU0sQ0FBQ0ksT0FBT0MsU0FBUyxHQUFHTCwrQ0FBUUEsQ0FBUztJQUMzQyxNQUFNLENBQUNNLFdBQVdDLGFBQWEsR0FBR1AsK0NBQVFBLENBQVU7SUFFcERELGdEQUFTQSxDQUFDO1FBQ1IsTUFBTVMsY0FBYyxJQUFJQyxZQUFZO1FBQ3BDRCxZQUFZRSxTQUFTLEdBQUcsQ0FBQ0M7WUFDckIsTUFBTSxDQUFDQyxlQUFlQyxTQUFTLEdBQUdGLE1BQU1HLElBQUksQ0FBQ0MsS0FBSyxDQUFDO1lBQ25EWixjQUFjYSxXQUFXSjtZQUN6QlAsU0FBU1E7UUFDYjtRQUVBLE9BQU87WUFDSEwsWUFBWVMsS0FBSztRQUNyQjtJQUNGLEdBQUcsRUFBRTtJQUdMLE1BQU1DLGtCQUFrQjtRQUN0QixJQUFJO1lBQ0YsSUFBSVosV0FBVztnQkFDYix1Q0FBdUM7Z0JBQ3ZDQyxhQUFhO1lBQ2YsT0FBTztnQkFDTCxvQ0FBb0M7Z0JBQ3BDLE1BQU1ZLFdBQVcsTUFBTUMsTUFBTSx5Q0FBeUM7b0JBQUVDLFFBQVE7Z0JBQU87Z0JBQ3ZGLE1BQU1QLE9BQU8sTUFBTUssU0FBU0csSUFBSTtnQkFDaENDLFFBQVFDLEdBQUcsQ0FBQ1Y7Z0JBQ1pQLGFBQWE7WUFDZjtRQUNGLEVBQUUsT0FBT2tCLE9BQU87WUFDZEYsUUFBUUUsS0FBSyxDQUFDLDZCQUE2QkE7UUFDN0M7SUFDRjtJQUVBLHFCQUNFLDhEQUFDQzs7MEJBQ0MsOERBQUNDOzBCQUFHOzs7Ozs7MEJBQ0osOERBQUNEOztrQ0FDQyw4REFBQ0U7OzRCQUFFOzRCQUFhMUIsV0FBVzJCLE9BQU8sQ0FBQzs7Ozs7OztrQ0FDbkMsOERBQUNEOzs0QkFBRTs0QkFBU3hCOzs7Ozs7O2tDQUNaLDhEQUFDMEI7d0JBQVNDLEtBQUs7d0JBQUdDLE9BQU85Qjs7Ozs7Ozs7Ozs7OzBCQUUzQiw4REFBQytCO2dCQUFPQyxTQUFTaEI7MEJBQ2RaLFlBQVksbUJBQW1COzs7Ozs7Ozs7Ozs7QUFJeEM7R0FqRE1MO0tBQUFBO0FBbUROLCtEQUFlQSxTQUFTQSxFQUFDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vX05fRS8uL3BhZ2VzL2luZGV4LnRzeD8wN2ZmIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VFZmZlY3QsIHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuXG5jb25zdCBJbmRleFBhZ2U6IFJlYWN0LkZDID0gKCkgPT4ge1xuICBjb25zdCBbY29uZmlkZW5jZSwgc2V0Q29uZmlkZW5jZV0gPSB1c2VTdGF0ZTxudW1iZXI+KDAuMCk7XG4gIGNvbnN0IFtsYWJlbCwgc2V0TGFiZWxdID0gdXNlU3RhdGU8c3RyaW5nPihcIkZhbHNlXCIpO1xuICBjb25zdCBbcmVjb3JkaW5nLCBzZXRSZWNvcmRpbmddID0gdXNlU3RhdGU8Ym9vbGVhbj4oZmFsc2UpO1xuICBcbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBjb25zdCBldmVudFNvdXJjZSA9IG5ldyBFdmVudFNvdXJjZSgnaHR0cDovL2xvY2FsaG9zdDo4MDgwL2F1ZGlvX3N0cmVhbScpO1xuICAgIGV2ZW50U291cmNlLm9ubWVzc2FnZSA9IChldmVudCkgPT4ge1xuICAgICAgICBjb25zdCBbbmV3Q29uZmlkZW5jZSwgbmV3TGFiZWxdID0gZXZlbnQuZGF0YS5zcGxpdCgnLCcpO1xuICAgICAgICBzZXRDb25maWRlbmNlKHBhcnNlRmxvYXQobmV3Q29uZmlkZW5jZSkpO1xuICAgICAgICBzZXRMYWJlbChuZXdMYWJlbCk7XG4gICAgfTtcblxuICAgIHJldHVybiAoKSA9PiB7XG4gICAgICAgIGV2ZW50U291cmNlLmNsb3NlKCk7XG4gICAgfTtcbiAgfSwgW10pO1xuXG5cbiAgY29uc3QgdG9nZ2xlUmVjb3JkaW5nID0gYXN5bmMgKCkgPT4ge1xuICAgIHRyeSB7XG4gICAgICBpZiAocmVjb3JkaW5nKSB7XG4gICAgICAgIC8vIElmIGFscmVhZHkgcmVjb3JkaW5nLCBzdG9wIHJlY29yZGluZ1xuICAgICAgICBzZXRSZWNvcmRpbmcoZmFsc2UpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLy8gSWYgbm90IHJlY29yZGluZywgc3RhcnQgcmVjb3JkaW5nXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgZmV0Y2goJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4MC9zdGFydF9yZWNvcmRpbmcnLCB7IG1ldGhvZDogJ1BPU1QnIH0pO1xuICAgICAgICBjb25zdCBkYXRhID0gYXdhaXQgcmVzcG9uc2UuanNvbigpO1xuICAgICAgICBjb25zb2xlLmxvZyhkYXRhKTtcbiAgICAgICAgc2V0UmVjb3JkaW5nKHRydWUpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICBjb25zb2xlLmVycm9yKCdFcnJvciB0b2dnbGluZyByZWNvcmRpbmc6JywgZXJyb3IpO1xuICAgIH1cbiAgfTtcblxuICByZXR1cm4gKFxuICAgIDxkaXY+XG4gICAgICA8aDE+QXVkaW8gQ2xhc3NpZmljYXRpb24gRGVtbzwvaDE+XG4gICAgICA8ZGl2PlxuICAgICAgICA8cD5Db25maWRlbmNlOiB7Y29uZmlkZW5jZS50b0ZpeGVkKDIpfTwvcD5cbiAgICAgICAgPHA+RGFuZ2VyOiB7bGFiZWx9PC9wPlxuICAgICAgICA8cHJvZ3Jlc3MgbWF4PXsxfSB2YWx1ZT17Y29uZmlkZW5jZX0+PC9wcm9ncmVzcz5cbiAgICAgIDwvZGl2PlxuICAgICAgPGJ1dHRvbiBvbkNsaWNrPXt0b2dnbGVSZWNvcmRpbmd9PlxuICAgICAgICB7cmVjb3JkaW5nID8gJ1N0b3AgUmVjb3JkaW5nJyA6ICdTdGFydCBSZWNvcmRpbmcnfVxuICAgICAgPC9idXR0b24+XG4gICAgPC9kaXY+XG4gICk7XG59O1xuXG5leHBvcnQgZGVmYXVsdCBJbmRleFBhZ2U7XG4iXSwibmFtZXMiOlsiUmVhY3QiLCJ1c2VFZmZlY3QiLCJ1c2VTdGF0ZSIsIkluZGV4UGFnZSIsImNvbmZpZGVuY2UiLCJzZXRDb25maWRlbmNlIiwibGFiZWwiLCJzZXRMYWJlbCIsInJlY29yZGluZyIsInNldFJlY29yZGluZyIsImV2ZW50U291cmNlIiwiRXZlbnRTb3VyY2UiLCJvbm1lc3NhZ2UiLCJldmVudCIsIm5ld0NvbmZpZGVuY2UiLCJuZXdMYWJlbCIsImRhdGEiLCJzcGxpdCIsInBhcnNlRmxvYXQiLCJjbG9zZSIsInRvZ2dsZVJlY29yZGluZyIsInJlc3BvbnNlIiwiZmV0Y2giLCJtZXRob2QiLCJqc29uIiwiY29uc29sZSIsImxvZyIsImVycm9yIiwiZGl2IiwiaDEiLCJwIiwidG9GaXhlZCIsInByb2dyZXNzIiwibWF4IiwidmFsdWUiLCJidXR0b24iLCJvbkNsaWNrIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./pages/index.tsx\n"));

/***/ })

},
/******/ function(__webpack_require__) { // webpackRuntimeModules
/******/ var __webpack_exec__ = function(moduleId) { return __webpack_require__(__webpack_require__.s = moduleId); }
/******/ __webpack_require__.O(0, ["pages/_app","main"], function() { return __webpack_exec__("./node_modules/next/dist/build/webpack/loaders/next-client-pages-loader.js?absolutePagePath=C%3A%5CUsers%5Cdevli%5CDocuments%5CDevelopment%5Cmhacks%5Cwebsite%5Cclient%5Cpages%5Cindex.tsx&page=%2F!"); });
/******/ var __webpack_exports__ = __webpack_require__.O();
/******/ _N_E = __webpack_exports__;
/******/ }
]);