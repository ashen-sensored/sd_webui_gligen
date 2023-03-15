import launch

if not launch.is_installed("natsort"):
    launch.run_pip("install natsort", "requirements for gligen")

if not launch.is_installed("easing_functions"):
    launch.run_pip("install easing_functions", "requirements for gligen")
