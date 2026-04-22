/**
 * babel.config.js — NexusLock
 *
 * IMPORTANT:
 * • babel-preset-expo (SDK 55) already includes the Reanimated / Worklets
 *   babel plugin, so we do NOT add react-native-reanimated/plugin here.
 * • react-native-worklets-core/plugin is required for VisionCamera
 *   frame-processor worklets — it must be listed explicitly.
 */
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
    plugins: [["react-native-worklets-core/plugin"]],
  };
};
