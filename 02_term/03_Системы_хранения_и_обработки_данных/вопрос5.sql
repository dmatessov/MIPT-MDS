explain query plan

select group_st.num_group, count(*)
from group_st3
join group_st on group_st.id = group_st3.id
where group_st.age between 18 and 27 and 
    (group_st3.score > 3 or group_st.num_group = 3)
group by group_st.num_group
having count(*) > 1
order by group_st.num_group