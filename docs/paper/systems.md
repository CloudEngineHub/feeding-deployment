\section{System}
\label{sec:system}

We deploy a mobile mealtime-assistance robot in the CR's home that autonomously executes the full mealtime pipeline --- from retrieving a meal from the refrigerator to returning the plate to the sink. Building on FEAST~\cite{jenamani2025feast}, which demonstrated personalized feeding assistance across in-the-wild settings, we extend the system with a mobile base for autonomous navigation throughout the home and new skills spanning the full mealtime pipeline: meal retrieval from the refrigerator, heating in the microwave, placing the meal on the table and feeding, and cleanup at the sink. To our knowledge, this is the first mealtime-assistance system capable of end-to-end pipeline execution; prior work universally assumes the plate is already on the table~\cite{jenamani2025feast, jenamani2024flair, jenamani2024bitetransfer, nanavati2025lessons, bhattacharjee2020moreautonomy, park2020evaluation, ha2024repeat}. We describe the hardware, navigation, manipulation, levels of autonomy, and web interface.

\subsection{Hardware}
\label{subsec:hardware}

\textbf{Mobile Base.}
The robot arm is mounted on a Vention~\cite{vention} stand fitted with differential-drive wheels, enabling autonomous navigation throughout the home. Two RPLidar~A1~\cite{rplidar} sensors mounted on the stand provide 360° 2D lidar for simultaneous localization and mapping. A ZED Mini~\cite{zed} stereo camera on the stand provides depth sensing for obstacle avoidance. An Arduino Uno~\cite{arduino} on the stand receives velocity commands from the compute laptop and routes them to the base motors.

\textbf{Robot Arm.}
The arm is a Kinova Gen3~\cite{kinova} 7-DoF manipulator with a Robotiq 2F-85~\cite{robotiq} gripper. An RGB-D Intel RealSense~\cite{camera} camera and a 6-axis ATI Nano25~\cite{ati} force/torque sensor on the feeding utensil are used for manipulation perception and contact detection.

\textbf{Tool Apparatus.}
The Vention stand carries a plate holder, a drink holder, and a tool stand. The CR's standard plate is fitted with a custom \emph{plate attachment} --- a 3D-printed polypropylene clip that gives the gripper a rigid, repeatable grasp surface and docks the plate into the stand's plate holder between tasks. Polypropylene is microwave-safe, so the plate can be placed directly in the microwave without removing the attachment. Similarly, the CR's mug is fitted with a custom \emph{drink attachment} that docks into the drink holder on the stand. The tool stand holds the feeding and wiping utensils between uses. The \emph{feeding utensil} integrates wrist-like motorized degrees of freedom enabling twirling, scooping, and upright food transport; a metal fork is connected to the ATI Nano25~\cite{ati} force/torque sensor to detect bite completion. The \emph{wiping utensil} holds a removable microfiber cloth for mouth wiping.

\textbf{Emergency Stop.}
Each emergency stop connects via a physical wire to a portable device. One is wired to the CR's iPad, through which she also uses the web interface. The other is wired to a handheld device with the researcher. Routing to portable devices rather than fixed positions accommodates the mobile base.

\textbf{Compute and Networking.}
The primary compute platform is a Lenovo Legion Pro~7i laptop~\cite{compute} with a 16GB RTX 4090 GPU, handling perception, planning, and preference learning. A dedicated Intel NUC~\cite{nuc} runs joint-space and task-space controllers for the arm on a separate system from the main laptop, enabling safety watchdogs that can halt the arm independently of main compute. Communication between the laptop, NUC, robot arm, web interface, and base Arduino is managed via a Nighthawk RAX43 router~\cite{router}. All compute components are mounted on the Vention stand and move with the base.

\subsection{Navigation}
\label{subsec:navigation}

During the preparation phase, we build a 2D occupancy map of the CR's home using Google Cartographer~\cite{hess2016real} with the two RPLidar~A1 sensors and record named waypoints for the four task locations: refrigerator, microwave, dining table, and sink.

During deployment, the ZED Mini provides continuous visual-inertial odometry (VIO), while Cartographer fuses the RPLidar~A1 scans to perform loop closure and correct accumulated drift. This combination achieves localization accurate to approximately 10~cm in position and 0.1~radians in heading. ROS \texttt{move\_base}~\cite{move_base}, using the Timed Elastic Band (TEB) local planner~\cite{rosmann2013efficient} for non-holonomic trajectory optimization, plans collision-free paths to named waypoints, with velocity commands routed to the base motors via the Arduino. Navigation between task locations is a precondition for all manipulation skills, so the PDDL planner automatically inserts navigation steps as needed.

\subsection{Manipulation}
\label{subsec:manipulation}

Manipulation skills are implemented as parameterized behavior trees~\cite{colledanchise2018behavior}, each exposing typed parameters (e.g., $\texttt{Speed} \in \{\texttt{slow, medium, fast}\}$) whose values are adapted from CR corrections. A PDDL task planner (FastDownward~\cite{fd}) sequences skills into valid plans given the current propositional state. The full skill library for this deployment covers the mealtime pipeline:

\begin{itemize}[leftmargin=*, labelindent=0pt]
    \item \texttt{OpenDoor} / \texttt{CloseDoor}: Opens or closes the refrigerator or microwave door.
    \item \texttt{PressMicrowaveButton}: Presses buttons to set and start a heating cycle.
    \item \texttt{PickPlate} / \texttt{PlacePlate}: Picks up or sets down the meal plate; sources and destinations include the fridge, microwave, table, base holder, and sink.
    \item \texttt{PickTool} / \texttt{StowTool}: Picks up or returns the feeding or wiping utensil from the tool stand.
    \item \texttt{AcquireBite}: Selects and acquires a bite using the food manipulation skill library~\cite{jenamani2024flair}.
    \item \texttt{TransferTool}: Delivers a bite, sip, or mouth wipe using head-pose-based tracking~\cite{jenamani2024bitetransfer}.
\end{itemize}

Skills that require scene understanding use vision-language foundation models for perception. GroundingDINO~\cite{liu2023grounding} performs open-set object detection to localize task-relevant objects such as door handles and the sink basin; SAM~\cite{kirillov2023segment} refines these into pixel-accurate masks. For fine-grained point localization --- specifically, identifying the microwave start button --- we use MolmoPoint-7B~\cite{deitke2024molmo} prompted with natural language. Detections are back-projected through the RealSense depth map to yield 3D keypoints that parameterize each skill, which are then executed via Cartesian end-effector or joint-space controllers on the NUC.

\subsection{Levels of Autonomy}
\label{subsec:autonomy}

The system supports multiple levels of autonomy. In \emph{full autonomy}, the robot executes the mealtime pipeline end-to-end; the CR monitors and may correct a preference or trigger recovery if needed. In \emph{semi-autonomous} operation, the CR teleoperates the mobile base via the web interface --- for example, repositioning it to resolve a joint limit during door manipulation. In \emph{full teleoperation}, the CR can control both the base and the arm directly via the web interface. As the CR develops a model of the robot's failure modes over the deployment, she may begin to proactively reposition the base before a skill begins rather than reacting after failure --- itself a form of user co-adaptation.

\subsection{User Interface}
\label{subsec:interface}

Similar to FEAST~\cite{jenamani2025feast}, the web interface is implemented in Vue.js~\cite{vue}, runs on the CR's iPad, and communicates with the robot via ROS. It retains all prior pages: a \emph{New Meal Page} for specifying food items and bite order, a \emph{Task Selection Page} for requesting bites, sips, or mouth wipes, a \emph{Bite Acquisition Page} for reviewing the selected bite, a \emph{Manual Bite Acquisition Page} for specifying keypoint parameters directly, and a \emph{Personalization Page} for language-based adaptations and transparency queries.

For this deployment we add two pages. The \emph{Pre-Meal Preferences Page}, shown at the start of each meal, displays the system's predicted values for all preference dimensions alongside available options; the CR can correct any dimension, and the system re-predicts the full bundle incorporating those corrections. The \emph{Teleoperation Page} allows the CR to drive the mobile base and command the arm directly, supporting semi-autonomous and fully teleoperated operation.
