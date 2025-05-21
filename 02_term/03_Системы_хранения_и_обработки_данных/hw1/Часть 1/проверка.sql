SELECT employee.id,
       employee.employee,
       supervisor.supervisor,
       employee.travel_required,
       department.department,
       position.position
  FROM employee AS employee
       JOIN
       department AS department ON employee.department = department.id
       JOIN
       supervisor AS supervisor ON employee.supervisor = supervisor.id
       JOIN
       position AS position ON employee.position = position.id;
