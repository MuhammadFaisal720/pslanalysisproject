import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:psl/themes/color.dart';
import 'package:http/http.dart' as http;

class TopTeamsPrediction extends StatefulWidget {
  @override
  _TopTeamsPredictionState createState() => _TopTeamsPredictionState();
}

class _TopTeamsPredictionState extends State<TopTeamsPrediction> {
  List users = [];
  bool isLoading = false;
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    this.fetchUser();
  }
  fetchUser() async {
    setState(() {
      isLoading = true;
    });
    var url = "pslapiversion1.herokuapp.com";
    var response = await http.get(Uri.https(url,'/topwinnerteam'));
    print(response.body);
    if(response.statusCode == 200){
      var items = json.decode(response.body);
      print(items);
      print(items[0]['Rank']);
      print(items[0]['Team']);
      print(items[0]['Score']);
      setState(() {
        users = items;
        isLoading = false;
      });
    }else{
      users = [];
      isLoading = false;
    }
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Winner Team Prediction"),
      ),
      body: getBody(),
    );
  }
  Widget getBody(){
    if(users.contains(null) || users.length < 0 || isLoading){
      return Center(child: CircularProgressIndicator(valueColor: new AlwaysStoppedAnimation<Color>(primary),));
    }
    return ListView.builder(
        itemCount: users.length,

        itemBuilder: (context,index){
          print(index);
          return getCard(users[index]);
        });
  }
  Widget getCard(item){
    print(item['Rank']);
    print(item['Team']);
    print(item['Score']);
    var ranking = "Rank:"+item['Rank'].toString();
    var teamname="Team:"+item['Team'].toString();
    var score = "Score:"+item['Score'].toString();
    //var profileUrl = item['picture']['large'];
    return Card(
      elevation: 1.5,
      child: Padding(
        padding: const EdgeInsets.all(10.0),
        child: ListTile(
          title: Row(
            children: <Widget>[
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  SizedBox(
                      width: MediaQuery.of(context).size.width-140,
                      child: Text(ranking,style: TextStyle(fontSize: 17),)),
                  SizedBox(
                      width: MediaQuery.of(context).size.width-140,
                      child: Text(teamname,style: TextStyle(fontSize: 17),)),
                  SizedBox(height: 10,),
                  Text(score,style: TextStyle(color: Colors.grey),)
                ],
              )
            ],
          ),
        ),
      ),
    );
  }
}