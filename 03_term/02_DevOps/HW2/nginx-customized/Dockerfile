# Use the official Nginx image as the base image
FROM nginx

# Remove the default Nginx welcome page
RUN rm -f /usr/share/nginx/html/index.html

# Copy your custom index.html to the Nginx html directory
COPY index.html /usr/share/nginx/html/

# Expose port 80 for Nginx
EXPOSE 80

# The CMD instruction sets the default command to run when a container starts
CMD ["nginx", "-g", "daemon off;"]
