# whsun-viz

Web application for hosting the Sun Group visualizations at the University of Michigan.
https://whsunresearch.group/viz

## Testing

This assumes you have [Git](https://git-scm.com/install/) and [Git LFS](https://git-lfs.com/) installed on your machine.

First install [Docker](https://www.docker.com/) for your system.

Fork then clone the repository to your local.

Build the Docker image with `docker build -t whsun-viz src` from the repository folder.

Run the application with `docker run -it -p 8000:8000 whsun-viz`. You can test the application by navigating to http://localhost:8000/. `Ctrl + C` will shut down the application.

Once tested, submit a PR for approval.

## Modifying

Create a new folder under `src/app` with the name of your visualization. This is where you can put the code for you visualizations. 

Modify the `main.py` file to mount your application.

If your application or code requires any additional dependencies, you can append them to the `requirements.txt` file.
